"""Launching distributed graph partitioning pipeline """
import os
import sys
import argparse
import logging
import json

INSTALL_DIR = os.path.abspath(os.path.join(__file__, '..'))
LAUNCH_SCRIPT = "distgraphlaunch.py"
PIPELINE_SCRIPT = "distpartitioning/data_proc_pipeline.py"

UDF_WORLD_SIZE = "world-size"
UDF_PART_DIR = "partitions-dir"
UDF_INPUT_DIR = "input-dir"
UDF_GRAPH_NAME = "graph-name"
UDF_SCHEMA = "schema"
UDF_NUM_PARTS = "num-parts"
UDF_OUT_DIR = "output"

LARG_PROCS_MACHINE = "num_proc_per_machine"
LARG_IPCONF = "ip_config"
LARG_MASTER_PORT = "master_port"

def get_launch_cmd(args) -> str:
    cmd = sys.executable + " " + os.path.join(INSTALL_DIR, LAUNCH_SCRIPT)
    cmd = f"{cmd} --{LARG_PROCS_MACHINE} 1 "
    cmd = f"{cmd} --{LARG_IPCONF} {args.ip_config} "
    cmd = f"{cmd} --{LARG_MASTER_PORT} {args.master_port} "

    return cmd


def submit_jobs(args) -> str:
    wrapper_command = os.path.join(INSTALL_DIR, LAUNCH_SCRIPT)

    #read the json file and get the remaining argument here.
    schema_path = "metadata.json"
    with open(os.path.join(args.in_dir, schema_path)) as schema:
        schema_map = json.load(schema)

    num_parts = len(schema_map["num_nodes_per_chunk"][0])
    graph_name = schema_map["graph_name"]

    argslist = ""
    argslist += "--world-size {} ".format(num_parts)
    argslist += "--partitions-dir {} ".format(os.path.abspath(args.partitions_dir))
    argslist += "--input-dir {} ".format(os.path.abspath(args.in_dir))
    argslist += "--graph-name {} ".format(graph_name)
    argslist += "--schema {} ".format(schema_path)
    argslist += "--num-parts {} ".format(num_parts)
    argslist += "--output {} ".format(os.path.abspath(args.out_dir))

    # (BarclayII) Is it safe to assume all the workers have the Python executable at the same path?
    pipeline_cmd = os.path.join(INSTALL_DIR, PIPELINE_SCRIPT)
    udf_cmd = f"{args.python_path} {pipeline_cmd} {argslist}"

    launch_cmd = get_launch_cmd(args)
    launch_cmd += '\"'+udf_cmd+'\"'

    print(launch_cmd)
    os.system(launch_cmd)

def main():
    parser = argparse.ArgumentParser(description='Dispatch edge index and data to partitions', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-dir', type=str, help='Location of the input directory where the dataset is located')
    parser.add_argument('--partitions-dir', type=str, help='Location of the partition-id mapping files which define node-ids and their respective partition-ids, relative to the input directory')
    parser.add_argument('--out-dir', type=str, help='Location of the output directory where the graph partitions will be created by this pipeline')
    parser.add_argument('--ip-config', type=str, help='File location of IP configuration for server processes')
    parser.add_argument('--master-port', type=int, default=12345, help='port used by gloo group to create randezvous point')
    parser.add_argument('--python-path', type=str, default=sys.executable, help='Path to the Python executable on all workers')

    args, udf_command = parser.parse_known_args()

    assert os.path.isdir(args.in_dir)
    assert os.path.isdir(args.partitions_dir)
    assert os.path.isfile(args.ip_config)
    assert isinstance(args.master_port, int)

    tokens = sys.executable.split(os.sep)
    submit_jobs(args)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
