from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="star_melt",  
    version="0.9.0",  
    author="Justyn Campbell-White",
    author_email="astrojustyn@gmail.com",  
    description="STAR MELT package",  
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justyncw/star_melt",  
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  
        install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "astropy",
        "qgrid",
        "ipywidgets",
        "ipython",
        "PyAstronomy",
        "astroquery",
        "lmfit",
    ],
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        'star_melt': ['Line_Resources/*']
    }
)