# created a ML Projrct folder

# Creating a virtual environment
# in windows is
python --version to check python version
python -m ven ven_name
python -m venv mlpro to creat a virtul environment
# Action the virtual environment
<!-- ./ven_names/Scrits/activate -->
./mlpro/Scripts/activate
# increase you get an error while activating virtual environment
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

pwd - present working directory
cd - to move to a specific directory
ls - to list all files

#to install all libraries from requirements file 
cd backend
pip install -r requirments.txt
cd .. #come out of backend

# git init
# git add backend/ frontend/ README.MD
# git commit -m "v1"

git remote add origin https://github.com/mujeebr/Ml_Project_Bidar.git
git branch -M main
git push -u origin main


# from second time

git add modified files
# git commit -m "v1"
git push -u origin main
