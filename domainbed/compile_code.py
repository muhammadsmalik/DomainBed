import os

def compile_code(base_path, output_file='domainbed_code_compilation.txt'):
    """
    Compiles code from specified files into a single text file.
    
    Args:
        base_path (str): Base path to the DomainBed project
        output_file (str): Name of the output file
    """
    # Define the files we want to compile (in order)
    files_to_compile = [
        # Core implementation files
        ('algorithms.py', 'Core Algorithms'),
        ('datasets.py', 'Dataset Implementations'),
        ('networks.py', 'Neural Network Architectures'),
        ('hparams_registry.py', 'Hyperparameter Registry'),
        ('model_selection.py', 'Model Selection'),
        
        # Scripts
        ('scripts/train.py', 'Training Script'),
        ('scripts/sweep.py', 'Hyperparameter Sweep'),
        ('scripts/collect_results.py', 'Results Collection'),
        
        # Library files
        ('lib/fast_data_loader.py', 'Fast Data Loader'),
        ('lib/misc.py', 'Utility Functions'),
        ('lib/wide_resnet.py', 'Wide ResNet Implementation')
    ]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("DomainBed Code Compilation\n")
        outfile.write("=" * 50 + "\n\n")
        
        for file_path, section_title in files_to_compile:
            full_path = os.path.join(base_path, file_path)
            
            # Write section header
            outfile.write(f"\n{section_title}\n")
            outfile.write("-" * len(section_title) + "\n")
            outfile.write(f"File: {file_path}\n")
            outfile.write("=" * 50 + "\n\n")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n")
                    outfile.write("=" * 50 + "\n\n")
            except FileNotFoundError:
                outfile.write(f"ERROR: File not found: {file_path}\n\n")
            except Exception as e:
                outfile.write(f"ERROR: Could not read file {file_path}: {str(e)}\n\n")

def main():
    # Assuming you're running this from the project root
    # Replace this path with the actual path to your DomainBed directory
    base_path = "."  # or provide the full path to the domainbed directory
    output_file = "domainbed_code_compilation.txt"
    
    compile_code(base_path, output_file)
    print(f"Code compilation complete. Check {output_file}")

if __name__ == "__main__":
    main()