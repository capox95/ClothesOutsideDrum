#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloudIntT;

class Processing
{
private:
    pcl::ModelCoefficients::Ptr plane_;
    PointCloudT::Ptr scene_;
    PointCloudT::Ptr object_result_;
    PointCloudIntT::Ptr scene_map_;

public:
    Processing() : plane_(new pcl::ModelCoefficients),
                   object_result_(new PointCloudT),
                   scene_(new PointCloudT),
                   scene_map_(new PointCloudIntT)
    {
    }

    void compute(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg);

    void setSceneCloud(PointCloudT::Ptr scene);

    void setObjectCloud(PointCloudT::Ptr object);

    pcl::ModelCoefficients::Ptr getPlaneUsed();

    void visualize();

private:
    void
    getPointPlaneDistanceCloud(PointCloudT::Ptr &obj,
                               PointCloudT::Ptr &scene,
                               PointCloudT::Ptr scene_proj,
                               PointCloudIntT::Ptr &cloud_out,
                               pcl::ModelCoefficients::Ptr &plane);

    void computeDiffProjected(PointCloudT::Ptr obj,
                              PointCloudT::Ptr scene,
                              PointCloudIntT::Ptr cloud_output,
                              pcl::ModelCoefficients::Ptr &plane);

    void segmentSceneCloud(PointCloudIntT::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg);
};