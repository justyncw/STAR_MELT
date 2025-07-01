from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="star_melt",  # Replace with your package name
    version="0.9.0",  # Match the version in your __init__.py
    author="Justyn Campbell-White",
    author_email="astrojustyn@gmail.com",  # Replace with your email
    description="STAR MELT package",  # Replace with your package description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justyncw/star_melt",  # Replace with your repository URL
    packages=find_packages(),  # Automatically find all sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Specify the minimum Python version
    install_requires=[
        # Add your dependencies here, e.g.:
        "numpy",
        "matplotlib",
        "pandas",
    ],
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        'star_melt': ['Line_Resources/*']
    }
)