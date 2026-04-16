# **RXMesh Documentation**  

[![Deploy](https://github.com/Ahdhn/RXMeshDocs/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ahdhn/RXMeshDocs/actions/workflows/deploy.yml)


The documentation is hosted at [ahdhn.github.io/RXMeshDocs](https://ahdhn.github.io/RXMeshDocs)

## Build (locally) 

On Windows or Linux, first install Conda, then run:

```
git clone https://github.com/Ahdhn/RXMeshDocs.git 
cd RXMeshDocs
conda env create -f environment.yml
conda activate RXMeshDocs
mkdocs serve 
```

Then open the displayed link (usually [http://127.0.0.1:8000/RXMeshDocs](http://127.0.0.1:8000/RXMeshDocs)) in your browser. This allows you to edit the files and see changes immediately.
