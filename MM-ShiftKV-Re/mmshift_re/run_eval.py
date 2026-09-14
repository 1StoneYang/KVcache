from mmshift_re.install import apply


def main() -> None:
    apply()
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
