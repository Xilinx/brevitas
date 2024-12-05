import yaml

from brevitas_examples.llm.main import parse_args

if __name__ == "__main__":
    default_args = parse_args([])
    with open('default_template.yml', 'w') as f:
        yaml.dump(default_args.__dict__, f)
