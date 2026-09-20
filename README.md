## Installation

> **Note:** This installation guide was generated with ChatGPT and may contain mistakes or platform-specific edge cases.
I do not understand the structure well enough to give detailed instructions, therefore genAI made these. Windows comments by me as I tested this step
### Windows

```command prompt  
git clone <repository-url>
cd P-project-based-physics-lab-

#create environment to edit all files locally
python -m venv .venv
.venv\Scripts\activate.bat

#make the files available to other python files inside that scope
#only need to be rerun once pyproject.toml gets changed
python -m pip install -e .


Linux:
(you should be smart enough to figure out errors:) )
git clone <repository-url>
cd P-project-based-physics-lab-

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e .


MacOs:

git clone <repository-url>
cd P-project-based-physics-lab-

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e .

