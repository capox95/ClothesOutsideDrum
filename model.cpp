#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>

// Types
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

int main(int argc, char **argv)
{
    // Point clouds
    PointCloudT::Ptr source(new PointCloudT);

    // Get input object and scene
    if (argc != 2)
    {
        pcl::console::print_error("Syntax is: %s object.pcd\n", argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<PointT>(argv[1], *source) < 0)
    {
        pcl::console::print_error("Error loading object/scene file!\n");
        return (1);
    }

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<PointT> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(source);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(source);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_plane);

    // Create the filtering object PassThrough

    PointCloudT::Ptr cloud_filtered(new PointCloudT);
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud_plane);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-1, 1.05);
    //pass.setFilterLimitsNegative (true);
    pass.filter(*cloud_filtered);

    PointCloudNT::Ptr cloud_normals(new PointCloudNT);

    cloud_normals->width = cloud_filtered->width;
    cloud_normals->height = cloud_filtered->height;
    cloud_normals->resize(cloud_normals->width * cloud_normals->height);
    for (int i = 0; i < cloud_filtered->size(); i++)
    {
        cloud_normals->points[i].x = cloud_filtered->points[i].x;
        cloud_normals->points[i].y = cloud_filtered->points[i].y;
        cloud_normals->points[i].z = cloud_filtered->points[i].z;
    }

    // Estimate normals for object
    pcl::NormalEstimationOMP<PointNT, PointNT> nestObj;
    nestObj.setNumberOfThreads(4);
    nestObj.setRadiusSearch(0.01);
    nestObj.setInputCloud(cloud_normals);
    nestObj.compute(*cloud_normals);

    pcl::visualization::PCLVisualizer viz("Visualizer");
    viz.addCoordinateSystem(0.1, "coordinate");
    viz.setBackgroundColor(0.0, 0.0, 0.5);
    viz.addPointCloud<PointT>(source, "source");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 1.0f, 0.0f, "source");
    viz.addPointCloud<PointNT>(cloud_normals, "cloud_plane");
    viz.spin();

    pcl::io::savePCDFileASCII("crop.pcd", *cloud_normals);

    return (0);
}
