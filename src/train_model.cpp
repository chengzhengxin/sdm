#include <vector>
#include <iostream>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include "modelcfg.h"
#include "ldmarkmodel.h"

using namespace std;
using namespace cv;


int main()
{
    std::vector<ImageLabel> mImageLabels;
    if(!load_ImageLabels("mImageLabels-train.bin", mImageLabels)){
        mImageLabels.clear();
        ReadLabelsFromFile(mImageLabels);
        save_ImageLabels(mImageLabels, "mImageLabels-train.bin");
    }
    std::cout << "训练数据一共有: " <<  mImageLabels.size() << std::endl;


    vector<vector<int>> LandmarkIndexs;
    vector<int> LandmarkIndex1(IteraLandmarkIndex1, IteraLandmarkIndex1+LandmarkLength1);
    LandmarkIndexs.push_back(LandmarkIndex1);
    vector<int> LandmarkIndex2(IteraLandmarkIndex2, IteraLandmarkIndex2+LandmarkLength2);
    LandmarkIndexs.push_back(LandmarkIndex2);
    vector<int> LandmarkIndex3(IteraLandmarkIndex3, IteraLandmarkIndex3+LandmarkLength3);
    LandmarkIndexs.push_back(LandmarkIndex3);
    vector<int> LandmarkIndex4(IteraLandmarkIndex4, IteraLandmarkIndex4+LandmarkLength4);
    LandmarkIndexs.push_back(LandmarkIndex4);
    vector<int> LandmarkIndex5(IteraLandmarkIndex5, IteraLandmarkIndex5+LandmarkLength5);
    LandmarkIndexs.push_back(LandmarkIndex5);

    vector<int> eyes_index(eyes_indexs, eyes_indexs+4);
    Mat mean_shape(1, 2*LandmarkPointsNum, CV_32FC1, mean_norm_shape);
    //vector<HoGParam> HoGParams{{ VlHogVariant::VlHogVariantUoctti, 5, 11, 4, 1.0f },{ VlHogVariant::VlHogVariantUoctti, 5, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 5, 8, 4, 0.4f },{ VlHogVariant::VlHogVariantUoctti, 5, 6, 4, 0.25f } };
    vector<HoGParam> HoGParams{{ VlHogVariant::VlHogVariantUoctti, 4, 11, 4, 0.9f },{ VlHogVariant::VlHogVariantUoctti, 4, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 4, 9, 4, 0.5f },{ VlHogVariant::VlHogVariantUoctti, 4, 8, 4, 0.3f }, { VlHogVariant::VlHogVariantUoctti, 4, 6, 4, 0.2f } };
    vector<LinearRegressor> LinearRegressors(5);

    ldmarkmodel model(LandmarkIndexs, eyes_index, mean_shape, HoGParams, LinearRegressors);
    model.train(mImageLabels);
    save_ldmarkmodel(model, "PCA-SDM-model.bin");


    system("pause");
    return 0;
}
