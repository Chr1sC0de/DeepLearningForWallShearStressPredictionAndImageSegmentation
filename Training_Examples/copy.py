import pathlib as pt
import shutil


if __name__ == "__main__":
    cwd = pt.Path(__file__).parent
    desired_folder = pt.Path(r"I:\CNNForCFD")
    
    for folder in desired_folder.glob('KFold*'):
        if folder.is_dir():
            config = next(folder.glob("config.py"))
            train = next(folder.glob("train.py"))
            
            my_fold = cwd/folder.name
            
            if not my_fold.exists():
                my_fold.mkdir()
                
            with open(config,'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if "data_folder" in line: 
                    lines[i] = "data_folder = current_directory/'../../../Data/NewtonianSteadyWSS'"
                    
            with open(my_fold/'config.py', 'w') as f:
                f.writelines(lines)
                
            with open(train,'r') as f:
                lines = f.readlines()
                
            with open(my_fold/'train.py', 'w') as f:
                f.writelines(lines)
                
            print('done')
            