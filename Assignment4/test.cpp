#include "precomp.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mapshift.hpp"

namespace cv {
namespace reg {

MapperGradShift::MapperGradShift()
{
}

MapperGradShift::~MapperGradShift()
{
}

cv::Ptr<Map> MapperGradShift::calculate(
    InputArray _img1, InputArray image2, cv::Ptr<Map> init) const
{
    Mat img1 = _img1.getMat();
    Mat gradx, grady, imgDiff;
    Mat img2;

    CV_DbgAssert(img1.size() == image2.size());

    if(!init.empty()) {
        // We have initial values for the registration: we move img2 to that initial reference
        init->inverseWarp(image2, img2);
    } else {
        img2 = image2.getMat();
    }

    // Get gradient in all channels
    gradient(img1, img2, gradx, grady, imgDiff);

    // Calculate parameters using least squares
    Matx<double, 2, 2> A;
    Vec<double, 2> b;

    A(0, 0) = sum(sum(gradx.mul(gradx)))[0];
    A(0, 1) = sum(sum(gradx.mul(grady)))[0];
    A(1, 1) = sum(sum(grady.mul(grady)))[0];
    A(1, 0) = A(0, 1);

    b(0) = -sum(sum(imgDiff.mul(gradx)))[0];
    b(1) = -sum(sum(imgDiff.mul(grady)))[0];

    // Calculate shift. We use Cholesky decomposition, as A is symmetric.
    Vec<double, 2> shift = A.inv(DECOMP_CHOLESKY)*b;

    if(init.empty()) {
        return Ptr<Map>(new MapShift(shift));
    } else {
        Ptr<MapShift> newTr(new MapShift(shift));
        MapShift* initPtr = dynamic_cast<MapShift*>(init.get());
        Ptr<MapShift> oldTr(new MapShift(initPtr->getShift()));
        oldTr->compose(newTr);
        return oldTr;
    }
}

cv::Ptr<Map> MapperGradShift::getMap() const
{
    return cv::Ptr<Map>(new MapShift());
}