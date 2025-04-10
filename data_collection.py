import os
from git import Repo


def clone_repositories(repos_list, target_dir):
    """Clone GitHub repositories to a local directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for repo_url in repos_list:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(target_dir, repo_name)

        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url} to {repo_path}")
            Repo.clone_from(repo_url, repo_path)
        else:
            print(f"Repository {repo_name} already exists")


def extract_python_files(repos_dir):
    """Extract all Python files from cloned repositories."""
    python_files = []
    for root, dirs, files in os.walk(repos_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files