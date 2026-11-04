from glob import glob

from setuptools import find_packages, setup

package = "pallet_rl_server"

setup(
    name=package,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/config", glob("config/*.yaml")),
        (f"share/{package}/launch", glob("launch/*.py")),
        (f"share/{package}/models", glob("models/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Koby Livne",
    maintainer_email="koby_livne@pickommerce.com",
    description="Stateless ROS 2 placement server for the pallet RL policy",
    license="Proprietary",
    entry_points={"console_scripts": [f"server = {package}.node:main"]},
)
