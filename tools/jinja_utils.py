import jinja2
import os

env = None

def jinja_init(rootdir):
    global env
    env = jinja2.Environment(
	loader=jinja2.FileSystemLoader(os.path.join(rootdir, "templates")),
	autoescape=jinja2.select_autoescape()
    )
    env.trim_blocks = True


def jinja_render(template, **kwargs):
    global env
    return env.get_template(template).render(**kwargs)

jinja_init(os.path.dirname(__file__))
