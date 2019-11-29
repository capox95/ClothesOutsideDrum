#include "alignProcessing.h"

void Processing::setSceneCloud(PointCloudT::Ptr scene) { scene_ = scene; }

void Processing::setObjectCloud(PointCloudT::Ptr object) { object_result_ = object; }

pcl::ModelCoefficients::Ptr Processing::getPlaneUsed() { return plane_; }

PointCloudIntT::Ptr Processing::getSceneMap() { return scene_map_; }

void Processing::getPointPlaneDistanceCloud(PointCloudT::Ptr &obj, PointCloudT::Ptr &scene, PointCloudT::Ptr scene_proj,
                                            PointCloudIntT::Ptr &cloud_out, pcl::ModelCoefficients::Ptr &plane)
{
    float threshold = 0.05;

    Eigen::Vector3f x0, y0;
    Eigen::Vector3f n(plane->values[0], plane->values[1], plane->values[2]);
    float p = plane->values[3];
    float distance, distance_obj;

    cloud_out->width = scene->width;
    cloud_out->height = scene->height;
    cloud_out->resize(scene->width * scene->height);

    pcl::search::KdTree<PointNT> kdtree;
    PointNT searchPoint;
    kdtree.setInputCloud(scene_proj);

    int K = 1;
    std::vector<int> indices(K);
    std::vector<float> distances(K);

    for (int i = 0; i < obj->size(); i++)
    {
        searchPoint = obj->points[i];

        if (kdtree.nearestKSearch(searchPoint, K, indices, distances) > 0)
        {
            if ((sqrt(distances[0]) < threshold) &&
                (!std::isnan(scene->points[indices[0]].x)) &&
                (!std::isnan(scene->points[indices[0]].y)) && (!std::isnan(scene->points[indices[0]].z)))
            {
                // computation for scene cloud
                x0 = scene->points[indices[0]].getVector3fMap();
                distance = n.dot(x0) + p;

                // computation for obj cloud
                y0 = obj->points[i].getVector3fMap();
                distance_obj = n.dot(y0) + p;

                // new cloud
                cloud_out->points[i].x = scene->points[indices[0]].x;
                cloud_out->points[i].y = scene->points[indices[0]].y;
                cloud_out->points[i].z = scene->points[indices[0]].z;

                if (distance_obj - distance > 0)
                    cloud_out->points[i].intensity = distance_obj - distance;
                else
                    cloud_out->points[i].intensity = 0;
            }
        }
        indices.clear();
        distances.clear();
    }
}

void Processing::computeDiffProjected(PointCloudT::Ptr obj, PointCloudT::Ptr scene,
                                      PointCloudIntT::Ptr cloud_output, pcl::ModelCoefficients::Ptr &plane)
{
    PointCloudT::Ptr obj_proj(new PointCloudT);
    PointCloudT::Ptr scene_proj(new PointCloudT);

    pcl::SACSegmentation<PointNT> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(0.05);
    seg.setInputCloud(obj);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    //std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Create the filtering object
    pcl::ProjectInliers<PointNT> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(scene);
    proj.setModelCoefficients(coefficients_plane);
    proj.filter(*scene_proj);

    getPointPlaneDistanceCloud(obj, scene, scene_proj, cloud_output, coefficients_plane);

    plane = coefficients_plane;
}

void Processing::segmentSceneCloud(PointCloudIntT::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg)
{
    int counter = 0;
    for (PointT point : cloud->points)
        if (point.intensity > 0.01)
            counter++;

    std::cout << counter << std::endl;

    cloud_seg->width = counter;
    cloud_seg->height = 1;
    cloud_seg->resize(cloud_seg->width);

    pcl::PointXYZRGB pointTmp;
    for (int i = 0; i < cloud->size(); i++)
    {
        if (cloud->points[i].intensity > 0.01)
        {
            pointTmp.x = cloud->points[i].x;
            pointTmp.y = cloud->points[i].y;
            pointTmp.z = cloud->points[i].z;
            cloud_seg->points.push_back(pointTmp);
        }
    }
}

void Processing::compute(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg)
{
    computeDiffProjected(object_result_, scene_, scene_map_, plane_);

    segmentSceneCloud(scene_map_, cloud_seg);
}

void Processing::visualize()
{
    //
    pcl::visualization::PCLVisualizer viz("Visualizer");
    //viz.addCoordinateSystem(0.1, "coordinate");
    viz.setBackgroundColor(0.0, 0.0, 0.5);

    viz.addPointCloud<PointNT>(scene_, "scene");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 1.0f, 0.0f, "scene");
    viz.addPointCloud<PointNT>(object_result_, "object_result");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "object_result");

    pcl::visualization::PCLVisualizer vizScene("Scene Map");
    vizScene.setBackgroundColor(0.0, 0.0, 0.5);
    //vizScene.addPointCloud<PointT>(diff_scene, "diff_scene");
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(scene_map_, "intensity");
    vizScene.addPointCloud<pcl::PointXYZI>(scene_map_, intensity_distribution, "diff_scene");
}