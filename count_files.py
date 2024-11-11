import os
from typing import Optional

def print_directory_tree(start_path: str, prefix: str = "", level: int = -1, exclude_dirs: Optional[list] = None) -> None:
    """
    Print the directory structure as a tree.
    
    Args:
        start_path (str): The root directory to start from
        prefix (str): Prefix for the current item (used for recursion)
        level (int): Maximum depth to traverse (-1 for unlimited)
        exclude_dirs (list): List of directory names to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', '.venv']
    
    # Check if we've reached the maximum depth
    if level == 0:
        return

    # Get and sort directory contents
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    except Exception as e:
        print(f"{prefix}[Error: {str(e)}]")
        return

    files = []
    directories = []

    # Separate files and directories
    for item in items:
        item_path = os.path.join(start_path, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path) and item not in exclude_dirs:
            directories.append(item)

    # Print files
    for i, f in enumerate(files):
        if i == len(files) - 1 and len(directories) == 0:
            print(f"{prefix}└── {f}")
        else:
            print(f"{prefix}├── {f}")

    # Print directories and their contents
    for i, d in enumerate(directories):
        is_last = i == len(directories) - 1
        print(f"{prefix}{'└── ' if is_last else '├── '}[{d}]")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        new_path = os.path.join(start_path, d)
        print_directory_tree(new_path, new_prefix, level - 1 if level > 0 else -1, exclude_dirs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Print directory structure as a tree")
    parser.add_argument("path", nargs="?", default=".", help="Path to the directory to print (default: current directory)")
    parser.add_argument("-l", "--level", type=int, default=-1, help="Maximum depth to traverse (-1 for unlimited)")
    parser.add_argument("-e", "--exclude", nargs="+", help="Additional directories to exclude")
    
    args = parser.parse_args()
    
    # Combine default excluded directories with user-provided ones
    exclude_dirs = ['.git', '__pycache__', 'node_modules', '.venv']
    if args.exclude:
        exclude_dirs.extend(args.exclude)
    
    print(f"\nDirectory structure for: {os.path.abspath(args.path)}\n")
    print_directory_tree(args.path, level=args.level, exclude_dirs=exclude_dirs)
    print()  # Add newline at the end