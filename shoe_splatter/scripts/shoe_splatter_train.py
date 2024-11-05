import os
import yaml
import argparse
from pathlib import Path
import utils


# placehodler training function
def train(input_path: Path) -> None:
    """
    Train shoe splatter model using nerfstudio.

    Args:
        input_path (Path): Path to the shoe_splatter results workspace.

    Returns:
        None: No return value.
    """
    images_path = input_path / "images"
    masks_path = input_path / "masks"
    output_path = input_path / "output"
    transforms_path = input_path / "transforms.json"

    print(f"Training shoe splatter data in {input_path}")
    utils.clear_and_create_directory(output_path)

    # Add masks to transforms.json
    # utils.add_mask_paths_to_transform(transforms_path)

    # Get the directory of the current script
    current_script_dir = Path(__file__).resolve().parent
    # Navigate to the config file relative to the script location
    config_path = current_script_dir.parents[1] / "configs" / "train.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration file and print its contents
    try:
        config = utils.read_yaml_config(config_path)
        print("Configuration loaded successfully:")
        for key, value in config.items():
            print(f"{key}: {value}")
    # Handle exceptions that may occur during configuration loading
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")

    # Parse the configuration and append additional arguments
    # base_command = f"ns-train splatfacto --data {str(train_path.resolve())} \
    #         --output-dir {str(output_path.resolve())}"
    # extra_commands = f"colmap --masks-path {str(masks_path.resolve())} \
    #         --images-path {str(images_path.resolve())}"

    base_command = f"ns-train splatfacto --output-dir {str(output_path.resolve())}"

    # base_command = f"ns-train splatfacto \
    #         --data {str(input_path.resolve())} \
    #         --output-dir {str(output_path.resolve())}"

    # --masks-path {str(masks_path.resolve())} \


    extra_commands =  f"colmap \
            --data {str(input_path.resolve())} \
            --images-path {str(images_path.resolve())} \
            --colmap-path {str((input_path/'colmap'/'sparse'/'masked').resolve())} \
            --masks-path {str(masks_path.resolve())}"
    command = utils.parse_config(base_command, config, extra_commands)
    # Run the command using the run_command function
    utils.run_command(command, current_script_dir.parents[1])


def main(args: argparse.Namespace) -> None:
    print("Shoe splatter model training started...")
    train(args.input)
    print("Shoe splatter model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train shoe_splatter model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the shoe_splatter results workspace.")
    args = parser.parse_args()
    main(args)