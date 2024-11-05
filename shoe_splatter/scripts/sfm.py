import os
import yaml
import argparse
from pathlib import Path
import utils


def process_data(input_path: Path, output_path: Path) -> None:
    """
    Process shoe splatter data.

    Args:
        input_path (Path): Path to the shoe_splatter results workspace.
        output_path (Path): Path to save the output results.

    Returns:
        None: No return value.
    """
    # Get the directory of the current script
    current_script_dir = Path(__file__).resolve().parent

    input_type, _ = utils.detect_input_type(input_path)

    # Navigate to the config file relative to the script location
    config_path = current_script_dir.parents[1] / "configs" / f"process_{input_type}.yml"
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

    # create output directory
    utils.clear_and_create_directory(output_path)

    # Parse the configuration and append additional arguments
    base_command = f"ns-process-data {input_type} --data {str(input_path.resolve())} --output-dir {str(output_path.resolve())}"
    command = utils.parse_config(base_command, config)

    # Run the command using the run_command function
    utils.run_command(command, current_script_dir.parents[1])


def main(args: argparse.Namespace) -> None:
    print("Shoe splatter data processing started...")
    process_data(args.input, args.output)
    print("Shoe splatter data processing completed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sfm for shoe_splatter model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the shoe_splatter images or video.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the output results in workspace")
    args = parser.parse_args()
    main(args)