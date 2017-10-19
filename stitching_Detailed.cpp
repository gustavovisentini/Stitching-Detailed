#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <typeinfo>

#include "opencv2/core/ocl.hpp"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "orb";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
std::string stitchingParamsFileName = "ParamsStitching";
string warp_type = "plane";
float match_conf = 0.3f;
string result_name = "result.jpg";
int range_width = -1;

//function to get warped and registered image
//takes input image, mask, corner point, roi, roi mask, output image reference, output image mask reference
void getWarpedRegisteredImage(InputArray _img, InputArray _mask, Point tl, Rect dst_roi_, Mat dst_mask_, Mat dst_)
{
    int corner_x, corner_y;
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    
    //update the corner points for each image
    corner_x = tl.x - dst_roi_.x;
    corner_y = tl.y - dst_roi_.y;
    
    //update output image and image mask with corners updates
    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(corner_y + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(corner_y + y);
        
        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[corner_x + x] = src_row[x];
            dst_mask_row[corner_x + x] |= mask_row[x];
        }
    }
}

vector<Mat> getStitchingParams(vector<Mat> InputImage)
{
#if 0
    cv::setBreakOnError(true);
#endif
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    
    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif
    
    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            finder = makePtr<SurfFeaturesFinderGpu>();
        else
#endif
            finder = makePtr<SurfFeaturesFinder>();
    }
    else if (features_type == "orb")
    {
        finder = makePtr<OrbFeaturesFinder>();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
    }
    
    int num_images = (int)InputImage.size();
    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;
    
    for (int i = 0; i < num_images; ++i)
    {
        full_img = (Mat)InputImage.at(i);
        full_img_sizes[i] = full_img.size();
        
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        
        (*finder)(img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());
        
        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }
    
    finder->collectGarbage();
    full_img.release();
    img.release();
    
    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    LOG("Pairwise matching..\n");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width==-1)
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
    
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();
    
    LOGLN("Pairwise matching- time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    
    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    
    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();
    
    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
    }
    
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }
    
    //Bundle Adjustment
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
    }
    
    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }
    
    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif
    
    std::ofstream out(stitchingParamsFileName.c_str(), std::ios::out);
    out << warped_image_scale;
    out.close();
    
    for (int i = 0; i < num_images; ++i)
    {
        stringstream camId;
        camId << i+1;
        string fileName = "cam" + camId.str() + ".yml";
        FileStorage fs(fileName, FileStorage::WRITE);
        fs << "K" << cameras[i].K();
        fs << "R" << cameras[i].R;
        fs << "t" << cameras[i].t;
        fs << "ppx" << cameras[i].ppx;
        fs << "ppy" << cameras[i].ppy;
        fs << "focal" << cameras[i].focal;
        fs << "aspect" << cameras[i].aspect;
        fs.release();
    }
    
    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);
    
    // Prepare images masks for each input image with same size as input image
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }
    
    // Warp images and their masks
    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }
    
    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
    }
    
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    
    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    
    LOGLN("Warping images - time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    
#if ENABLE_LOG
    t = getTickCount();
#endif
    
    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    
    vector<Mat> RegisteredImages, RegisteredImagesMask;
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        // Read image and resize it if necessary
        full_img = InputImage.at(img_idx);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            
            // Compute relative scales
            compose_work_aspect = compose_scale / work_scale;
            
            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);
            
            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;
                
                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }
                
                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();
        
        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        
        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        
        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        
        Rect dst_roi, dst_roi_;
        
        //create blank image and image mask which will store the warped image
        dst_roi = resultRoi(corners, sizes);
        Mat dst_, dst_mask_;
        dst_.create(dst_roi.size(), CV_16SC3);
        dst_.setTo(Scalar::all(0));
        dst_mask_.create(dst_roi.size(), CV_8U);
        dst_mask_.setTo(Scalar::all(0));
        dst_roi_ = dst_roi;
        //LOGLN("\nFinal Roi: " << dst_roi_ << "\n");
        
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
        
        //save warped and registered image for each camera and also update the new corner
        getWarpedRegisteredImage(img_warped_s, mask_warped, corners[img_idx], dst_roi_, dst_mask_, dst_);
        
        resize(dst_, dst_, Size(1700,920), compose_scale, compose_scale);
        resize(dst_mask_, dst_mask_, Size(1700,920), compose_scale, compose_scale);
        
        RegisteredImages.push_back(dst_);
        RegisteredImagesMask.push_back(dst_mask_);
    }
    
    //    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    //    {
    //        stringstream camId;
    //        camId << img_idx + 1;
    //        String warpedRegisteredImageFileName, warpedRegisteredImageMaskFileName;
    //        Mat stitchedImage = RegisteredImages[img_idx];
    //        Mat stitchedImageMask = RegisteredImagesMask[img_idx];
    //        warpedRegisteredImageFileName = "warpedRegisteredImage_" + camId.str() + ".jpg";
    //        warpedRegisteredImageMaskFileName = "warpedRegistered_ImageMask_" + camId.str() + ".jpg";
    //        imwrite(warpedRegisteredImageFileName, stitchedImage);
    //        imwrite(warpedRegisteredImageMaskFileName, stitchedImageMask);
    //    }
    //    LOGLN("\nFinished Analysis \n");
    
    return RegisteredImages;
}

int main(int argc, char* argv[])
{
    
    cv::ocl::setUseOpenCL(false);
    
    /*
    vector<Mat> inputImages, outputImages;
    String imageName;
    for (int i = 1; i < argc; ++i)
    {
        imageName = String(argv[i]);
        Mat img = imread(imageName);
        inputImages.push_back(img);
    }
     */
    
    cv::String path("/Users/gustavovisentini/Documents/Mestrado/AutoMosaicing/Phantom/*.JPG"); //select only jpg
    vector<cv::String> fn;
    vector<cv::Mat> imgs, outputImages;
    
    cv::glob(path,fn,true);
    
    cout << "Loading Images...\n\n";
    
    for (size_t k=0; k<fn.size(); ++k){
        cv::Mat im = cv::imread(fn[k]);
        if (im.empty()) continue;
        imgs.push_back(im);
    }
    
    // Check if have enough images
    int num_images = static_cast<int>(imgs.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    outputImages = getStitchingParams(imgs);
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        stringstream camId;
        camId << img_idx + 1;
        String warpedRegisteredImageFileName, warpedRegisteredImageMaskFileName;
        Mat stitchedImage = outputImages[img_idx];
        Mat stitchedImageMask = outputImages[img_idx];
        warpedRegisteredImageFileName = "warpedRegisteredImage_" + camId.str() + ".jpg";
        warpedRegisteredImageMaskFileName = "warpedRegistered_ImageMask_" + camId.str() + ".jpg";
        
        imwrite(warpedRegisteredImageFileName, stitchedImage);
        imwrite(warpedRegisteredImageMaskFileName, stitchedImageMask);
        
        
    }
    

    return 0;
}
