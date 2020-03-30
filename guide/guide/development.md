# Development

::: tip NOTE
The following is only necessary if you want to contribute features or
adversarial attacks to Foolbox. As a user of Foolbox, you can just do a normal
[installation](./getting-started).
:::

## Installation

First clone the repsository using `git`:

```bash
git clone https://github.com/bethgelab/foolbox
```

You can then do an editable installation using `pip -e`:

```bash
cd foolbox
pip3 install -e .
```

::: tip
Create a new branch for each new feature or contribution.
This will be necessary to open a pull request later.
:::

## Coding Style

We follow the [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
We use [black](https://github.com/psf/black) for automatic code formatting.
In addition, we use [flake8](https://flake8.pycqa.org/en/latest/) to detect
certain PEP 8 violations.

::: tip
Have a look at the `Makefile`. It contains many useful commands, e.g. `make black` or `make flake8`.
:::

## Type annotions and MyPy

Foolbox uses Python type annotations introduced in [PEP 484](https://www.python.org/dev/peps/pep-0484/).
We use [mypy](http://mypy-lang.org) for static type checking with relatively
strict settings. All code in Foolbox has to be type annotated.

We recommend to run MyPy or a comparable type checker automatically in your
editor (e.g. VIM) or IDE (e.g. PyCharm). You can also run MyPy from the
command line:

```bash
make mypy  # run this in the root folder that contains the Makefile
```

::: tip NOTE
`__init__` methods in Foolbox should not have return type annotations unless
they have no type annotated arguments (i.e. only `self`), in which case
the return type of `__init__` should be specifed as `None`.
:::

## Creating a pull request on GitHub

First, fork the [Foolbox repository on GitHub](https://github.com/bethgelab/foolbox).
Then, add the fork to your local GitHub repository:

```bash
git remote add fork https://github.com/YOUR USERNAME/foolbox
```

Finally, push your new branch to GitHub and open a pull request.
