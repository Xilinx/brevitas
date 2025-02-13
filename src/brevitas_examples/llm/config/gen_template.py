import yaml

from brevitas_examples.llm.main import parse_args

if __name__ == "__main__":
    default_args, _ = parse_args([])
    args_dict = default_args.__dict__
    del args_dict["config"]  # Config file cannot be specified via YAML
    with open('default_template.yml', 'w') as f:
        yaml.dump(args_dict, f)
