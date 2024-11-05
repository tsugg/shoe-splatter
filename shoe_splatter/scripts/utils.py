import sys
import yaml
import shutil
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Optional
from collections import defaultdict


def check_directories_and_files(input_path: Path) -> bool:
    """
    Check if the required directories exist and if file names are identical across all subdirectories.

    Args:
        input_path (Path): Path to the root directory to check.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    if not input_path.is_dir():
        print(f"Error: The input path {input_path} is not a directory.")
        return False

    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Error: No subdirectories found in {input_path}.")
        return False

    file_names = defaultdict(set)

    for subdir in subdirs:
        for file in subdir.iterdir():
            if file.is_file():
                file_names[subdir.name].add(file.name)

    if not file_names:
        print("Error: No files found in any subdirectory.")
        return False

    # Check if file names are identical across all subdirectories
    first_subdir = next(iter(file_names))
    reference_files = file_names[first_subdir]

    if all(set(files) == reference_files for files in file_names.values()):
        print(f"All subdirectories in {input_path} exist and contain identical file names.")
        return True
    else:
        print(f"Error: File names are not identical across all subdirectories in {input_path}.")
        return False


def run_command(command: str, cwd: Optional[str] = None) -> None:
    """
    Run a shell command, log its output live, and handle errors.

    Args:
        command (str): The command to run.
        cwd (Optional[str]): The working directory for the command. Defaults to None.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
    """
    print(f"Running command: {command}")
    print(f"Working directory: {cwd or 'current directory'}")

    try:
        # Use Popen to get live output
        with subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as process:
            # Read and log output line by line
            for line in process.stdout:
                print(line.strip())

        # Check the return code after the process completes
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Error output:\n{e.output}")
        print(f"Exception traceback:\n{traceback.format_exc()}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred while running the command: {str(e)}")
        print(f"Exception traceback:\n{traceback.format_exc()}")
        raise

    print("Command completed successfully")


def detect_input_type(input_path: Path) -> tuple[str, str]:
    """
    Detect if the input path is a video file or a directory of images.

    Args:
        input_path (Path): Path to the input file or directory.

    Returns:
        tuple[str, str]: A tuple containing the input type ('video' or 'image_dir') and the name.

    Raises:
        ValueError: If the input type is not supported (neither a video file nor a directory of images).
    """
    if input_path.is_file():
        # List of common video file extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if input_path.suffix.lower() in video_extensions:
            return 'video', input_path.stem.lower()
    elif input_path.is_dir():
        return 'images', input_path.parent.name
    else:
        raise ValueError(f"Unsupported input type: {input_path}. Expected a video file or a directory of images.")


def get_image_files(directory: Path) -> list[Path]:
    """
    Get a sorted list of all image files in the given directory.

    Args:
        directory (Path): Path to the directory containing image files.

    Returns:
        list[Path]: A sorted list of Path objects for each image file.
    """
    # Define common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Use a list comprehension to get all image files
    image_files = [
        file for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]
    
    # Sort the image files
    image_files.sort()
    
    return image_files


def clear_and_create_directory(directory: Path) -> None:
    """
    Check if a directory exists, delete it if it does, and create a new one.

    Args:
        directory (Path): Path to the directory to be cleared and created.

    Returns:
        None
    """
    if directory.exists() and directory.is_dir():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Cleared and created directory: {directory}")
    
    
def read_yaml_config(config_path: Path) -> dict:
    """
    Read a YAML configuration file and return its contents as a Python dictionary.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def parse_config(base_command: str, config: dict, additional_command: str = "") -> str:
    """
    Parse a configuration dictionary and return a single string command compatible with subprocess.run(),
    pre-appending a base command, adding '--' prefix to keys, and appending an additional command.

    Args:
        base_command (str): A string representing the base command (e.g., "python main.py").
        config (dict): A dictionary containing configuration settings without '--' prefixes.
        additional_command (str, optional): A string representing an additional command to be executed after the base command.

    Returns:
        str: A string representing the full command-line arguments.

    Example:
        Input:
        base_command = "python main.py"
        config = {
            "input": "input_file.txt",
            "output": "output_file.txt",
            "verbose": True,
            "level": 3
        }
        additional_command = "python post_process.py"

        Output:
        'python main.py --input input_file.txt --output output_file.txt --verbose --level 3 python post_process.py'
    """
    command_parts = [base_command]

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command_parts.append(f"--{key}")
        else:
            command_parts.extend([f"--{key}", str(value)])

    if additional_command:
        command_parts.append(additional_command)
    return " ".join(command_parts)


import json
from pathlib import Path

def add_mask_paths_to_transform(transform_path: Path) -> None:
    """
    Read the transform.json file, add a "mask_path" field to each frame,
    and save the updated JSON.

    Args:
        transform_path (Path): Path to the transform.json file.

    Raises:
        FileNotFoundError: If the transform.json file doesn't exist.
        json.JSONDecodeError: If there's an error parsing the JSON file.
    """
    if not transform_path.exists():
        raise FileNotFoundError(f"transform.json file not found: {transform_path}")

    try:
        with open(transform_path, 'r') as f:
            transform_data = json.load(f)

        for frame in transform_data['frames']:
            file_path = Path(frame['file_path'])
            file_name = file_path.name
            mask_path = Path("masks") / file_name
            frame['mask_path'] = str(mask_path)

        with open(transform_path, 'w') as f:
            json.dump(transform_data, f, indent=2)

        print(f"Updated {transform_path} with mask paths.")

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file: {e}", e.doc, e.pos)