# Import local modules from 'src/utils' as package 'utils'
import sys; sys.path.insert(0, '../')

import os
import utils
import socket
import ray

def attach_ray(manager: bool = False) -> None:
    """
    Start or connect the API to a ray instance.

    Note that a running API bind will be closed, if it is a standalone ray instance it will be fully terminated.

    :param manager: In case manager is set, trying to connect to the local instance with port 2099
    """

    disconnect_ray()

    if manager and not is_node_head():
        print("Caution: Assign node as manager different from project .env file.")

    if manager:
        ray.init(address='localhost:2099', runtime_env={ "py_modules": [utils] })
    else:
        ray.init(runtime_env={ "py_modules": [utils] })

    for node in get_cluster_nodes():
        print(node)

def disconnect_ray() -> None:
    """
    Wrapper function for `ray.shutdown()`. Testing for running API binds.
    """

    if ray.is_initialized():
        ray.shutdown()

def get_cluster_nodes():
    """
    Uses `ray.nodes()` to provide basic informations about all connected nodes.

    :return: Dict with the node hostname and resources informations.
    """

    for node in ray.nodes():
        if node['Alive'] or node['alive']:
            yield {
                'hostname': node['NodeManagerHostname'],
                'resources': node['Resources']
            }

def is_node_head(name: str = None) -> bool:
    """
    Determ if the given (given) host is the head node of the cluster.

    Note: If no name is provided, using `socket.gethostname()` to determ.

    :param name: Name to test against the enviorment config.
    :return:
    """
    if not name:
        name = socket.gethostname()
    return name == _read_cluster_config()

def _read_cluster_config() -> str:
    """
    Helper function for reading cluster enviorment variables.
    """

    return os.getenv("CLUSTER_HEAD_NAME")

