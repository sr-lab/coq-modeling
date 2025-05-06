import os
import sys
import csv
import time
import shutil
import logging
import argparse
import subprocess
from coqpyt.coq.base_file import CoqFile
from typing import List, Callable
from multiprocessing import Pool, cpu_count, TimeoutError


class ValidFile:
    def __init__(self, relpath: str, workspace: str):
        self.relpath: str = relpath
        self.workspace: str = workspace
        self.coq_version: str = "8.18"

    def __iter__(self):
        return iter([self.relpath, self.workspace, self.coq_version])


class Project:
    def __init__(self, dir_path: str, debug: bool = False):
        self.path: str = os.path.dirname(dir_path)
        self.found_valid_make: bool = False
        self.found_coqproject: bool = False
        self.valid_files: List[ValidFile] = []
        self.__log: Callable[[str], None] = logging.info if debug else lambda _: None
        
        self.__handle_dir(self.path)
        if not self.found_valid_make:
            self.__log(f"{self.name} has no valid Makefile")
        if not self.found_coqproject:
            self.__log(f"{self.name} has no _CoqProject")

    @property
    def name(self):
        return os.path.basename(self.path)

    def __relpath(self, path):
        return os.path.relpath(path, self.path)

    @staticmethod
    def __is_coq_file(file_path):
        return os.path.exists(file_path) and file_path.endswith(".v")

    @staticmethod
    def __is_valid(file_path, workspace, ran_make=False):
        try:
            with CoqFile(file_path, timeout=60, workspace=workspace) as state:
                if not ran_make:
                    state.save_vo()
                return state.is_valid
        except:
            return False

    def __execute_make(self, workspace):
        def get_valid_files(workspace, dir_path):
            for entry in os.listdir(dir_path):
                path = os.path.join(dir_path, entry)
                if Project.__is_coq_file(path) and Project.__is_valid(path, workspace, True):
                    valid_file = ValidFile(self.__relpath(path), workspace)
                    self.valid_files.append(valid_file)
                elif os.path.isdir(path):
                    get_valid_files(workspace, path)

        # Redirect stderr to stdout and ignore all output
        subprocess.check_output(f"make -C {workspace} 2>&1", shell=True)
        self.found_valid_make = True
        self.__log(f"{self.name} has valid Makefile in {self.__relpath(workspace)}")
        get_valid_files(workspace, workspace)

    def __parse_coqproject(self, workspace):
        self.found_coqproject = True
        self.__log(f"{self.name} has _CoqProject in {self.__relpath(workspace)}")
        with open(os.path.join(workspace, "_CoqProject"), "r") as f:
            for line in f.readlines():
                path = os.path.join(workspace, line.strip())
                if Project.__is_coq_file(path) and Project.__is_valid(path, workspace):
                    valid_file = ValidFile(self.__relpath(path), workspace)
                    self.valid_files.append(valid_file)

    def __handle_dir(self, dir_path):
        try:
            self.__execute_make(dir_path)
        except subprocess.CalledProcessError:
            if "_CoqProject" in os.listdir(dir_path):
                self.__parse_coqproject(dir_path)
                return

            # Filter dirs and Coq files
            dirs, files = [], []
            for entry in os.listdir(dir_path):
                path = os.path.join(dir_path, entry)
                if Project.__is_coq_file(path):
                    files.append(path)
                elif os.path.isdir(path):
                    dirs.append(path)

            # The order may vary depending on the filesystem. To ensure that
            # valid files don't change between runs on different machines, the
            # entries should be sorted in a specific order.
            # We check dirs first so that we can capture any potential
            # Makefile/_CoqProject before looking at the current files.
            for dir in sorted(dirs):
                self.__handle_dir(dir)
            for file in sorted(files):
                if Project.__is_valid(file, self.path):
                    valid_file = ValidFile(self.__relpath(file), self.path)
                    self.valid_files.append(valid_file)

    def export_csv(self):
        with open(os.path.join(self.path, "valid_files.csv"), "w") as f:
            csv.writer(f).writerows(self.valid_files)

    def delete(self):
        def retry_remove(function, path, excinfo):
            time.sleep(0.5)
            if os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path) and os.path.isfile(path):
                os.remove(path)

        shutil.rmtree(self.path, onerror=retry_remove)


def process_project(repo_path):
    csv_path = os.path.join(repo_path, "valid_files.csv")
    if os.path.exists(csv_path):
        print(f"Skipping {os.path.basename(repo_path)}: CSV already exists.")
        return
    try:
        project = Project(repo_path + "/")
        project.export_csv()
        print(f"Compiled {os.path.basename(repo_path)}...")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to process {repo_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos_loc", required=True, type=str, help="Location of the repositories."
    )
    parser.add_argument(
        "--n_workers", type=int, default=cpu_count(), help="Number of parallel workers (default: number of CPUs)."
    )
    args = parser.parse_args()

    repo_paths = [
        os.path.join(args.repos_loc, repo)
        for repo in os.listdir(args.repos_loc)
        if os.path.isdir(os.path.join(args.repos_loc, repo))
    ]

    TIMEOUT = 1200  # 20 minutes

    with Pool(args.n_workers) as pool:
        results = []
        for repo_path in repo_paths:
            results.append(pool.apply_async(process_project, (repo_path,)))

        for repo_path, result in zip(repo_paths, results):
            try:
                result.get(timeout=TIMEOUT)
            except TimeoutError:
                print(f"Timeout while processing {os.path.basename(repo_path)} (over 30 minutes).")
