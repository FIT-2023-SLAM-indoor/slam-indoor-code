import open3d as o3d

import config as cfg


def main():
    visualize_points_from_file(cfg.VIZ_FILE_PATH, cfg.VIZ_PARSE_FORMAT)


def visualize_points_from_file(file_path, parse_format="auto"):
    geometry = o3d.io.read_point_cloud(cfg.PROJECT_PATH + file_path, 
                                       format=parse_format)
    o3d.visualization.draw(geometry)


if __name__ == "__main__":
    main()
