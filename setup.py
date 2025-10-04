from setuptools import setup, find_packages

setup(
    name="foodclustering",
    version="1.0.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="Food clustering and health recommendation package",
    packages=find_packages(),
    include_package_data=True,  # ensures CSVs inside package are included
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires='>=3.8',
)
