# Setup

## Virtual environment

Activate the vitual environment _aip_project_ using the following command:

```console
$ source aip_project/bin/activate
```

## Requirements

To download the project requirements, run the following command:

```console
$ python3 -m pip install -r requirements.txt
```

To update the requirements file, run the following command:

```console
$ python3 -m pip freeze > 'requirements.txt'
```


# Running the scripts

Run the _main.py_ by using the following command:

```console
$ python3 main.py
```

You can change the input image in the top of the file in the main.py. Images can be displayed at different stages (just uncomment the image displaying function in the pipeline). All code is documented with pydocs. 
