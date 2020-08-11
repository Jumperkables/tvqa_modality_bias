import visdom
import numpy as np
import getpass

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        if getpass.getuser() == "jumperkables":
            self.viz = visdom.Visdom()
        self.env = env_name
        # Remove an enviroment if it exists
        if env_name in self.viz.get_env_list():
            self.viz.delete_env(env_name)
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    def text_plot(self, var_name, t):
        if var_name not in self.plots:
            self.plots[var_name]    = self.viz.text(text=t, env=self.env)
        else:
            self.viz.text(text=t, env=self.env, win=self.plots[var_name])
