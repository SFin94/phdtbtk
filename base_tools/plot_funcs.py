import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.lines as mlin

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from sklearn.preprocessing import LabelEncoder


'''General plotting functions - currently copied from mol lego version'''

def plot_setup(figsizeX=8, figsizeY=6, fig=None, ax=None):
    """
    Set general plot settings and initialise figure.

    Parameters:
     figsizeX: int - x dimension of plot [default: 12]
     figsizeY: int - y dimension of plot [default: 10]
     fig: matplotlib fig object - if other type of plot is called first [default: None]
     ax: matplotlib axes object - if other type of plot is called first [default: None]

    Returns:
     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
    
    """
    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    # colour_grey = '#3E3E3E'
    # plt.rcParams.update({'text.color': colour_grey, 'axes.labelcolor': colour_grey, 'xtick.color': colour_grey, 'ytick.color': colour_grey}) 

    # Set figure and plot param(s) vs energy
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsizeX,figsizeY))

    # Remove plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax


def radial_plot_setup(figsizeX=6, figsizeY=6, fig=None, ax=None):

    '''
    Function that sets some general settings for all plots

    Parameters:
     figsizeX: int - x dimension of plot [default: 12]
     figsizeY: int - y dimension of plot [default: 10]
     fig: matplotlib fig object - if other type of plot is called first [default: None]
     ax: matplotlib axes object - if other type of plot is called first [default: None]

    Returns:
     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
    '''

    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    colour_grey = '#3E3E3E'
    plt.rcParams.update({'text.color': colour_grey, 'axes.labelcolor': colour_grey, 'xtick.color': colour_grey, 'ytick.color': colour_grey}) 

    # Set figure and plot param(s) vs energy
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsizeX,figsizeY), subplot_kw=dict(projection='polar'))

    # ax.spines["circle"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax


def plot_mols_param(mol_data, param_col, save=None, colour=None, conf_labels=None, line=False):

    '''
    Plots parameter value vs. molecule/conformer as scatter plot

    Parameters:
     mol_data: pd DataFrame - contains the molecules and parameter values
     param_col: str - column heading to be plotted from the molecule DataFrame
     save: str - name to save plot image as (minus .png extension) [default: None type]
     colour: colour (matplotlib) - colour for plotting scatter points in [default: None type]. If default then a cubehelix colour is used.

    Returns:
     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
    '''

    # Set up plot
    fig, ax = plot_setup(figsizeX=8, figsizeY=7)

    # Set colour and line options
    if colour == None:
        colour = sns.cubehelix_palette(3, start=.5, rot=-.4, dark=0, light=0.5)[2]
    if line == None:
        line_style = ' '
    else:
        line_style = '--'

    # Plot figure and save if name given
    ax.plot(list(mol_data.index), mol_data[param_col], alpha=0.8, marker='o', color=colour, ls='--')
    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_mols_E(mol_data, energy_col=['Relative E'], save=None, colour=None, conf_labels=None, line=False):

    '''
    Function which plots molecules/conformers against the relative energy

    Parameters:
     mol_data: pandas DataFrame - Containing conformer names/keys and energies
     energy_col: str - header for relative energy column in dataframe [default: 'Relative E']
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     colour: colour (matplotlib) - colour to plot the conformers in [default: None type]. If default then a cubehelix colour is used.

    Returns:
     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plot_setup(figsizeX=8, figsizeY=7)

    # Plot conformer vx. relative energy
    if type(energy_col) != list:
        energy_col = [energy_col]
    if colour == None:
        colour = sns.cubehelix_palette(len(energy_col), start=.5, rot=-.4, dark=0, light=0.5)

    # Plot conformers for each quantity
    for col_ind, col in enumerate(energy_col):
        ax.scatter(list(mol_data.index), mol_data[col], marker='o', alpha=0.8, color=colour[col_ind], label=col, s=70)
        if line == True:
            ax.plot(list(mol_data.index), mol_data[col], alpha=0.3, color=colour[col_ind], ls='--')

    if conf_labels == None:
        conf_labels = mol_data.index

    # Set x and y labels and ticks
    ax.set_xticklabels(conf_labels, rotation=45)
    ax.set_ylabel('Relative Energy (kJmol$^{-1}$)', fontsize=13)
    ax.set_xlabel('Molecule', fontsize=13)
    plt.legend(fontsize=13)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_mols_all(mol_data, save=None, labels=None, enthalpy=False):

    '''
    Function which plots molecules against relative E and G (and H)

    Parameters:
     mol_data: pandas DataFrame - Containing conformer names/keys and energies
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     labels: list of str - list of molecule identifiers if different to data frame index [deafult: None type]

    Returns:
     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plot_setup()

    # Set colours for G and E
    e_colour = '#245F6B'
    g_colour = '#D17968'

    # Plot products vs. energy
    ax.scatter(mol_data.index, mol_data['Relative E'], color=e_colour, s=70, label='$\Delta$E')
    ax.scatter(mol_data.index, mol_data['Relative G'], color=g_colour, s=70, label='$\Delta$G')

    # Plot enthalpy (H) too if flag is True
    if enthalpy == True:
        h_colour = '#175443'
        ax.scatter(mol_data.index, mol_data['Relative H'], color=h_colour, s=70, label='$\Delta$H')

    # Set labels and axis settings
    if labels == None:
        labels = list(mol_data.index)

    ax.tick_params(labelsize=10)
    ax.set_xticklabels(labels, rotation=15, fontsize=11)
    ax.set_ylabel('Relative E/G (kJmol$^{-1}$)', fontsize=11)
    ax.set_xlabel('Molecule', fontsize=11)
    ax.legend(frameon=False, loc=1)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_param_E(mol_data, parameter_col, energy_col='Relative E SCF', save=None, colour=None, scan=False, fig=None, ax=None):
    """
    Plot molecules/conformers against the relative energy.

    Parameters
    ----------
    mol_data: pandas `DataFrame`
        Molecule data including conformer names/keys and energies.
    parameter_col: `str`
        Header for parameter column in molecule data.
    energy_col: `str`
        Header for relative energy column in dataframe [default: 'Relative E'].
    save: `str`
        Name of image to save plot too (minus .png extension) [Default: None].
    colour: `List of str`
        Hex Colour codes for colours to plot conformers in.
        [Default: None; sets colour for opt/uoptimised conformers].
    scan: `bool`
        If True then joins the scatterpoints with a line.
    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: NoneType]

    Returns
    -------
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    """
    fig, ax = plot_setup(fig=fig, ax=ax)

    # Set colours for plotting.
    if colour == None:
        # Set colours for unoptimised, optimised points.
        colour = ['#F26157', '#253237']
        if 'Optimised' in mol_data.columns.values:
            colour_list = []
            [colour_list.append(colour[opt]) for opt in mol_data['Optimised']]
            ax.legend(handles=[mlin.Line2D([], [], color=colour[0], label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), mlin.Line2D([], [], color=colour[1], label='Optimised', marker='o', alpha=0.6, linestyle=' ')], frameon=False, handletextpad=0.1, fontsize=10)
        else:
            colour_list = [colour[0]]*len(list(mol_data.index))

    # Make inputted colour argument the correct length for the number of conformers.
    elif len(colour) == len(list(mol_data.index)):
        colour_list = colour
    else:
        colour_list = [colour[0]]*len(list(mol_data.index))

    # Plot points and connecting lines if scan
    ax.scatter(mol_data[parameter_col], mol_data[energy_col], color=colour_list, marker='o', s=70, alpha=0.8)
    if scan == True:
        ax.plot(mol_data.sort_values(parameter_col)[parameter_col], mol_data.sort_values(parameter_col)[energy_col], marker=None, alpha=0.4, color=colour[1])

    # Set x and y labels
    ax.set_xlabel(parameter_col, fontsize=11)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=11)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_PES(mol_data, parameter_cols, energy_col='Relative E SCF', save=None, colour=None, opt_filter=True):

    '''
    Function which plots a 2D PES for two parameters

     Parameters:
      mol_data: pandas DataFrame - Containing conformer names/keys and energies
      paramOneCol: str - header for first parameter column in dataframe
      paramTwoCol: str - header for second parameter column in dataframe
      energy_col: str - header for relative energy column in dataframe [default: 'Relative E']
      save: str - name of image to save plot too (minus .png extension) [deafult: None type]
      colour: colour (matplotlib cmap) - colour map to plot the PES [default: None type; if default then a cubehelix colour map is used].

     Returns:
      fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

     '''

    fig, ax = plot_setup(figsizeX=7.5, figsizeY=6)

    # Filter out any unoptimised points if optimised present
    opt_col = ('Optimised' in mol_data.columns.values)
    if all([opt_filter, opt_col]):
        mol_data = mol_data[mol_data.Optimised]

    # Set linearly spaced parameter values and define grid between them
    param_one_range = np.linspace(mol_data[parameter_cols[0]].min(), mol_data[parameter_cols[0]].max(), 100)
    param_two_range = np.linspace(mol_data[parameter_cols[1]].min(), mol_data[parameter_cols[1]].max(), 100)
    param_one_grid, param_two_grid = np.meshgrid(param_one_range, param_two_range)

    # Interpolate the energy data on to the grid points for plotting
    interp_E = griddata((mol_data[parameter_cols[0]].values, mol_data[parameter_cols[1]].values), mol_data[energy_col], (param_one_grid, param_two_grid))

    # Set cmap if none provided
    if colour == None:
        colour = sns.cubehelix_palette(dark=0, as_cmap=True)

    # Plot filled contour and add colour bar
    c = ax.contourf(param_one_range, param_two_range, interp_E, 20, cmap=colour, vmax=150)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

    # Set x and y labels
    ax.set_xlabel(parameter_cols[0], fontsize=13)
    ax.set_ylabel(parameter_cols[1], fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def plot_reaction_profile(reaction_data, quantity_col='Relative G', save=None, colour=None, step_width=3000, line_buffer=0.08, label=True):

    '''Function which plots a reaction profile

    Parameters:
     reaction_data: pandas DataFrame
     energy_col: str - header for relative energy column in dataframe [default: 'Relative E']
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     colour: matplotlib cmap colour - colour map to generate path plot colours from [default: None type; if default then a cubehelix colour map is used].
     step_width: int - the marker size of the scatter hlines used to mark the reaction steps [default: 3000]
     line_buffer: float - the buffer from the centre of the hline that the connecting lines will connect from [default: 0.05]
     label: bool - if True then plots the indexes with each step, if False then returns the figure without labels

    Returns:
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plot_setup()
    paths = list(reaction_data['Reaction path'].unique())
    label_buffer = line_buffer - 0.01

    # Set colours if not provided - the number of paths will be number of colours
#    colours = sns.cubehelix_palette(len(paths))
    if colour == None:
        col_pallete = sns.color_palette("cubehelix", len(paths))
        colour = []
        for p_ind in range(len(paths)):
            colour.append(col_pallete[paths.index(p_ind)])

    # Plot the lines and points for the profile (line_buffer and step_width can be altered to fit the profile)
    for p_ind, path in enumerate(paths):
        reac_path_data = reaction_data.loc[reaction_data['Reaction path'] == path]
        ax.scatter(reac_path_data['Reaction coordinate'], reac_path_data[quantity_col], color=colour[p_ind], marker='_', s=step_width, lw=5)
        for rstep_ind in range(1, len(reac_path_data)):
            ax.plot([reac_path_data['Reaction coordinate'].iloc[rstep_ind-1]+line_buffer, reac_path_data['Reaction coordinate'].iloc[rstep_ind]-line_buffer], [reac_path_data[quantity_col].iloc[rstep_ind-1], reac_path_data[quantity_col].iloc[rstep_ind]],  color=colour[p_ind], linestyle='--')

            # Plot labels with dataframe index and energy label unless False, plot reactants at the end
            if label == True:
                step_label = reac_path_data.index.values[rstep_ind] + ' (' + str(int(reac_path_data[quantity_col].iloc[rstep_ind])) + ')'
                ax.text(reac_path_data['Reaction coordinate'].iloc[rstep_ind]-label_buffer, reac_path_data[quantity_col].iloc[rstep_ind]+6, step_label, color=colour[p_ind], fontsize=11)

        if label == True:
            reactant_label = reac_path_data.index.values[0] + ' (' + str(int(reac_path_data[quantity_col].iloc[0])) + ')'
            ax.text(reac_path_data['Reaction coordinate'].iloc[0]-label_buffer, reac_path_data[quantity_col].iloc[0]+6, reactant_label, color=colour[p_ind], fontsize=11)

    # Set x and y labels
    ax.set_xlabel('R$_{x}$', fontsize=13)
    ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
    ax.set_xticks([])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def normalise_parameters(conformer_data, geom_parameters):
    """
    Normalise geometric (bond/angle/dihedral) parameter values (all scaled to 0:1 range).

    Distances are normalised to [0:1] range.
    Angles are mapped from [0:180] range to [0:1] range.
    Dihedrals are mapped from [-180:180] range to [0:1] range.

    Updates DataFrame in place.

    Parameters
    ----------
    conformer_data: pandas `DataFrame`
        Conformer data containing values for geometric parameters to plot.
    geom_parameters: `list`
        Column headings defining the parameters.
    
    Returns
    -------
    param_headings: `list` of `str`
        parameter headings for the normalised parameters
    
    """
    param_headings = []
    for parameter in geom_parameters:
        if len(value) == 2:
            conformer_data["Norm " + key] = conformer_data[parameter]/conformer_data[parameter].max()
        elif len(value) == 3:
            conformer_data["Norm " + parameter] = conformer_data[parameter]/180.
        else:
            conformer_data["Norm " + parameter] = (conformer_data[parameter]%360.)/360.
        
        # Set parameter heading
        param_headings.append("Norm " + parameter)

    return param_headings


def set_conformer_colours(conformer_data, energy_col):

    '''Function that sets the colour for different conformers which can be normalised by energy values

    Parameters:
     conformer_data: pandas DataFrame - conformer data
     energy_col: str - name of the dataframe column header corresponding to the thermodynamic quantity to normalise the colours of the conformers too

    Returns:
     colblock/col_vals: list - colour code corresponding to each conformer 
    '''

    # Calculate normalised energy to plot colour by if given
    if energy_col != None:
        conformer_data['Norm E'] = conformer_data[energy_col]/conformer_data[energy_col].max()
        # colmap = sns.cubehelix_palette(start=2.5, rot=.5, dark=0, light=0.5, as_cmap=True)
        colmap = sns.cubehelix_palette(as_cmap=True)
        for val in conformer_data['Norm E']:
            colour_vals = [colmap(val)[:3] for val in conformer_data['Norm E']]    
        return colour_vals
    else:
    # Else set colours different for each conformer 
        colblock = sns.cubehelix_palette(len(conformer_data.index))
        return colblock

    
def plot_conf_radar(conformer_data, geom_parameters, save=None, colour=None, energy_col=None):
    """
    Plot conformers against several geometric parameters in a radial plot.

    Parameters
    ----------
    conformer_data: pandas `DataFrame`
        Conformer data containing values for geometric parameters to plot.
    geom_parameters: `list`
        Column headings defining the parameters.
    save: `str`
        Name of image to save plot too (minus .png extension) [deafult: None type].
    colour: :matplotlib:`cmap`
        Colour map to generate path plot colours from 
        [default: None type; if default then a cubehelix colour map is used].
    energy_col: `str`
        Column heading of optional energy parameter to colour rank conformers by.
        [default: None type]

    Returns:
     fig, ax - :matplotlib:`fig`, :matplotlib:`ax` objects for the plot

    """
    fig, ax = radial_plot_setup()

    # Calculate angles to plot, set parameter list
    num_params = len(geom_parameters)
    plot_angles = [n / float(num_params) * 2 * np.pi for n in range(num_params)]
    plot_angles += plot_angles[:1]

    # Normalise conformer parameters
    param_headings = normalise_parameters(conformer_data, geom_parameters)
    param_headings.append(param_headings[0])
    
    # Set colour
    if colour == None:
        conformer_data['Colour'] = set_conformer_colours(conformer_data, energy_col)
    else:
        conformer_data['Colour'] = colour

    # Plot for each conformer
    for conf in conformer_data.index:
        ax.plot(plot_angles, conformer_data.loc[conf][param_headings], label=conf, color=conformer_data.loc[conf]['Colour'])
        ax.fill(plot_angles, conformer_data.loc[conf][param_headings], color=conformer_data.loc[conf]['Colour'], alpha=0.1)

    # Set plot attributes
    ax.set_xticks(plot_angles[:-1])
    ax.set_xticklabels(geom_parameters)
    ax.set_yticks([])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def plot_conf_map(conformer_data, geom_parameters, save=None, colour=None, energy_col=None):
    """
    Plot conformers against several geometric parameters in a linear map.

    Parameters
    ----------
    conformer_data: pandas `DataFrame`
        Conformer data containing values for geometric parameters to plot.
    geom_parameters: `list`
        Column headings defining the parameters.
    save: `str`
        Name of image to save plot too (minus .png extension) [deafult: None type].
    colour: :matplotlib:`cmap`
        Colour map to generate path plot colours from 
        [default: None type; if default then a cubehelix colour map is used].
    energy_col: `str`
        Column heading of optional energy parameter to colour rank conformers by.
        [default: None type]

    Returns:
     fig, ax - :matplotlib:`fig`, :matplotlib:`ax` objects for the plot

    """
    fig, ax = plot_setup()
    num_params = len(geom_parameters)

    # Normalise conformer parameters
    param_headings = normalise_parameters(conformer_data, geom_parameters)

    # Set colour
    if colour == None:
        conformer_data['Colour'] = set_conformer_colours(conformer_data, energy_col)
    else:
        conformer_data['Colour'] = colour

    # Plot data
    for cInd, conf in enumerate(conformer_data.index):
        ax.plot(range(num_params), conformer_data.loc[conf][param_headings], label=conf, color=conformer_data.loc[conf]['Colour'], marker='o', alpha=0.8)

    # Set x and y labels and ticks
    ax.set_xticks(range(num_params))
    ax.set_xticklabels(plot_params, rotation=20, ha='right')
    ax.set_ylim(ymin=0.0, ymax=1.0)

    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # if energy_col != None:
    #     ax_cbar = inset_axes(ax, width="50%", height="3%", loc='upper right')
    #     plt.colorbar(cm.ScalarMappable(cmap=colmap), ax=ax, cax=ax_cbar, orientation="horizontal", ticks=[0, 1], label='$\Delta$G')

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def plot_order(mol_data_one, quantity_col_one, mol_data_two, quantity_col_two, save=None, fig=None, ax=None):
    """
    Plot comparative scatter plot of rankings for two sets of data.
    
    Parameters
    ----------
    mol_data_one: pandas `DataFrame`
        Conformer data containing quantity one values.
    quantity_col_one: `str`
        Column heading of quantity one to rank conformers by.
    mol_data_two: pandas `DataFrame`
        Conformer data (same conformers as mol_data_one) containing quantity two values.
    quantity_col_two: `str`
        Column heading of quantity two to rank conformers by.
    save: `str`
        Name of image to save plot too (minus .png extension) [deafult: None type].
    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: NoneType]
    
    Returns
    -------
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    """
    fig, ax = plot_setup(figsizeX=6, figsizeY=6, fig=fig, ax=ax)
    
    # Set order of molecules by first quantity
    mol_order = list((mol_data_one.sort_values(quantity_col_one).index))

    # Get rank of columns and compare - sorts by lowest first
    rank = []
    for mol_data, col in zip([mol_data_one, mol_data_two], [quantity_col_one, quantity_col_two]):
        order = list((mol_data.sort_values(col).index))
        rank.append([mol_order.index(i) for i in order])
    
    # Plot scatter of ranks
    ax.scatter(rank[0], rank[1], color='#5C6B73', s=150)

    # Set x and y axis as integer numbers
    conf_indexes = [int(i) for i in range(len(rank[0]))]
    ax.set_xticks(range(len(conf_indexes)))
    ax.set_yticks(range(len(conf_indexes)))
    ax.set_xticklabels(conf_indexes, fontsize=10)
    ax.set_yticklabels(conf_indexes, fontsize=10)

    if save != None:
        plt.savefig(save + '.png')
    
    return fig, ax


def plot_conf_comparison(mol_data_one, quantity_col_one, mol_data_two, quantity_col_two, labels, colours=['#9AD0BB','#5C6B73'], save=None):
    """
    Plot bar chart of relative quantities for two sets of conformers.
    
    Parameters
    ----------
    mol_data_one: pandas `DataFrame`
        Conformer data containing quantity one values.
    quantity_col_one: `str`
        Column heading of quantity one to plot.
    mol_data_two: pandas `DataFrame`
        Conformer data (same conformers as mol_data_one) containing quantity two values.
    quantity_col_two: `str`
        Column heading of quantity to plot.
    labels: `list of str`
        Labels for each of the conformer sets.
    save: `str`
        Name of image to save plot too (minus .png extension) [deafult: None type].
    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: NoneType]
    
    Returns
    -------
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    """
    # Initialise plot and variables.
    fig, ax = plot_setup()
    width=0.4
    x_range = np.arange(len(mol_data_one.index))

    # Plot bar chart for each conformer data set.
    ax.bar(x_range - width/2, mol_data_one[quantity_col_one], width, color=colours[0], label=labels[0], alpha=0.9)
    ax.bar(x_range + width/2, mol_data_two[quantity_col_two], width, color=colours[1], label=labels[1], alpha=0.9)

    # Set plot properties.
    ax.set_xticks(x_range)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)')
    ax.set_xlabel('Conformer')
    plt.legend()

    if save != None:
        plt.savefig(save + '.png')
    
    return fig, ax



### Plot Functions from molLego - 3rd Feb 2021

"""Module containing general plotting routines for molecules."""

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.lines as mlin
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

def plot_setup(figsize_x=8, figsize_y=6, fig=None, ax=None):
    """
    Initialise plot with general settings.

    Parameters
    ----------
    figsize_x : `int`
        x dimension of plot [default: 8]
    figsize_y : `int`
        y dimension of plot [default: 6]
    fig : :matplotlib:`fig`
        If other type of plot is called first [default: None]
    ax : :matplotlib:`axes`
        If other type of plot is called first [default: None]

    Returns
    -------
    fig, ax: :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
    """
    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    colour_grey = '#3E3E3E'
    plt.rcParams.update({'text.color': colour_grey, 
                         'axes.labelcolor': colour_grey, 
                         'xtick.color': colour_grey, 
                         'ytick.color': colour_grey})

    # Initiaise figure.
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsize_x,figsize_y))

    # Remove lines from plot frame.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax


def radial_plot_setup(figsize_x=6, figsize_y=6, fig=None, ax=None):
    """
    Initialise radial plot with general settings.

    Parameters
    ----------
    figsize_x : `int`
        x dimension of plot [default: 6]
    figsize_y : `int`
        y dimension of plot [default: 6]
    fig : :matplotlib:`fig`
        If other type of plot is called first [default: None]
    ax : :matplotlib:`axes`
        If other type of plot is called first [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
    """
    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    colour_grey = '#3E3E3E'
    plt.rcParams.update({'text.color': colour_grey, 
                         'axes.labelcolor': colour_grey, 
                         'xtick.color': colour_grey, 
                         'ytick.color': colour_grey})

    # Set figure and plot param(s) vs energy
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsize_x,figsize_y), 
                               subplot_kw=dict(projection='polar'))

    # ax.spines["circle"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax

def plot_mols_E(mol_data, energy_col=['Relative E'], save=None, 
                colour=None, mol_labels=None, line=False):
    """
    Plot molecules/conformers against relative energy.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    energy_col : str`
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]
    colour : `list of str`
        Colour list to plot molecules by.
        [default: None type; uses cubehelix colours].
    mol_labels : `list of str`
        Molecule identifiers if different to DatFrame index.
        [default: None type, uses DataFrame index]
    line : `bool`
        If True, connects scatterpoints by lines.
        
    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    fig, ax = plot_setup(figsize_x=8, figsize_y=7)

    # Plot conformer vs. relative energy
    if type(energy_col) != list:
        energy_col = [energy_col]
    if colour == None:
        colour = sns.cubehelix_palette(len(energy_col), start=.5, 
                                     rot=-.4, dark=0, light=0.5)

    # Plot conformers for each quantity
    for col_ind, col in enumerate(energy_col):
        ax.scatter(list(mol_data.index), mol_data[col], 
                        marker='o', alpha=0.8, color=[col_ind], 
                        label=col, s=70)
        if line == True:
            ax.plot(list(mol_data.index), mol_data[col], alpha=0.3, 
                         color=colour[col_ind], ls='--')

    if mol_labels == None:
        mol_labels = mol_data.index

    # Set x and y labels and ticks
    ax.set_xticklabels(mol_labels, rotation=15)
    ax.set_ylabel('Relative Energy (kJmol$^{-1}$)', fontsize=13)
    ax.set_xlabel('Molecule', fontsize=13)
    plt.legend(fontsize=13)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_mols_thermo(mol_data, save=None, mol_labels=None, enthalpy=False):
    """
    Plot molecules against relative E and G (and H).

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    mol_labels : `list of str`
        Molecule identifiers if different to DatFrame index.
        [default: None type, uses DataFrame index]
    enthalpy :
        If True also plots enthalpy values.
        [default: False]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()

    # Set colours for G and E
    e_colour = '#245F6B'
    g_colour = '#D17968'

    # Plot products vs. energy
    ax.scatter(mol_data.index, mol_data['Relative E'], 
               color=e_colour, s=70, label='$\Delta$E')
    ax.scatter(mol_data.index, mol_data['Relative G'], 
               color=g_colour, s=70, label='$\Delta$G')

    # Plot enthalpy (H) too if flag is True
    if enthalpy == True:
        h_colour = '#175443'
        ax.scatter(mol_data.index, mol_data['Relative H'], 
                   color=h_colour, s=70, label='$\Delta$H')

    # Set labels and axis settings
    if labels == None:
        labels = list(mol_data.index)

    ax.tick_params(labelsize=10)
    ax.set_xticklabels(labels, rotation=15, fontsize=11)
    ax.set_ylabel('Relative E/G (kJmol$^{-1}$)', fontsize=11)
    ax.set_xlabel('Molecule', fontsize=11)
    ax.legend(frameon=False, loc=1)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_param_E(mol_data, param_col, energy_col='Relative E', 
                 save=None, colour=None, scan=False):
    """
    Plot relative energies (or other specified quantity) of molecules.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    param_col : `str`
        Column header corresponding to geometric parameter to plot.
    energy_col : `str`
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure saved]
     colour : `str`
        A colour specified by a recognised matplotlib format.
        [default: None type; uses cubehelix colours].
     scan : `bool`
        If True, connects scatterpoints by lines.
        [default: False]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()

    # Set colours for plotting if not provided
    if colour == None:
        colour = ['#D17968', '#12304e']
    elif len(colour[0]) == 1:
        colour = [colour]

    # Set colours by molecule opt value.
    if 'Optimised' in mol_data.columns.values:
        colour_list = []
        [colour_list.append(colour[opt]) for opt in mol_data['Optimised']]
    elif len(colour) == len(list(mol_data.index)):
        colour_list = colour
    else:
        colour_list = [colour[0]]*len(list(mol_data.index))

    # Plot points and connecting lines if scan
    ax.scatter(mol_data[param_col], mol_data[energy_col], 
               color=colour_list, marker='o', s=70, alpha=0.8)
    if scan == True:
        ax.plot(mol_data[param_col], mol_data[energy_col], 
                marker=None, alpha=0.4, color=colour[1])

    # Set x and y labels
    ax.set_xlabel(param_col, fontsize=11)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=11)

    # Set legend to show unopt vs. opt points
    if 'Optimised' in mol_data.columns.values:
        ax.legend(handles=[mlin.Line2D([], [], color=colour[0], 
                  label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), 
                  mlin.Line2D([], [], color=colour[1], label='Optimised', 
                  marker='o', alpha=0.6, linestyle=' ')], frameon=False, 
                  handletextpad=0.1, fontsize=10)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_PES(mol_data, param_cols, energy_col='Relative E', 
             save=None, colour=None, opt_filter=True):
    """
    Plot 2D PES for two geometric parameters.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    param_cols : `list of str`
        Column headers corresponding to the two geometric parameter to plot.
    energy_col : ``str``
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : :matplotlib:`cmap`
        Colour map to plot PES. 
        [default: None type; uses cubehelix colour map].
    opt_filter : `bool`
        If True then removes unoptimised data points.
        [default: True]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup(figsize_x=7.5, figsize_y=6)

    # Filter out any unoptimised points if optimised present
    opt_col = ('Optimised' in mol_data.columns.values)
    if all([opt_filter, opt_col]):
        mol_data = mol_data[mol_data.Optimised]

    # Set linearly spaced parameter values and define grid between them
    param_one_range = np.linspace(mol_data[param_cols[0]].min(), 
                                  mol_data[param_cols[0]].max(), 100)
    param_two_range = np.linspace(mol_data[param_cols[1]].min(), 
                                  mol_data[param_cols[1]].max(), 100)
    param_one_grid, param_two_grid = np.meshgrid(param_one_range, 
                                                 param_two_range)

    # Interpolate the energy data on to the grid points for plotting
    interp_E = griddata((mol_data[param_cols[0]].values, 
                         mol_data[param_cols[1]].values), 
                         mol_data[energy_col], 
                         (param_one_grid, param_two_grid))

    # Set cmap if none provided
    if colour == None:
        colour = sns.cubehelix_palette(dark=0, as_cmap=True)

    # Plot filled contour and add colour bar
    c = ax.contourf(param_one_range, param_two_range, interp_E, 
                    20, cmap=colour, vmax=150)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

    # Set x and y labels
    ax.set_xlabel(param_cols[0], fontsize=13)
    ax.set_ylabel(param_cols[1], fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

def plot_reaction_profile(reaction_data, quantity_col='Relative G', save=None,
                            colour=None, step_width=3000, line_buffer=0.08, 
                            label=True, fig=None, ax=None):
    """
    Plot a reaction profile.

    Parameters
    ----------
    reaction_data : :pandas:`DataFrame`
        The reaction profile dataframe to plot.
    energy_col : ``str``
        Column header of quantity to plot reaction steps by.
        [default: 'Relative G']
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : :matplotlib:`cmap`
        Colour map to generate path plot colours from. 
        [default: None type; uses cubehelix colour map].
    step_width : `int`
        The marker size of the scatter hlines used to mark the reaction step.
        [default: 3000]
    line_buffer : `float`
        The buffer from centre of the hline of position
        the connecting lines will connect to.
        [default: 0.05]
    label : `bool`
        If True then plots the indexes with each step.
        If False then returns the figure without labels.
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup(fig=fig, ax=ax)
    paths = list(reaction_data['Reaction path'].unique())

    # Set colours if not provided.
    if colour == None:
        col_pallete = sns.color_palette("cubehelix", len(paths))
        colour = []
        for p_ind in range(len(paths)):
            colour.append(col_pallete[paths.index(p_ind)])

    # Plot lines and points for the profile.
    for p_ind, path in enumerate(paths):
        reac_path_data = reaction_data.loc[
                            reaction_data['Reaction path'] == path]
        ax.scatter(reac_path_data['Rx'], reac_path_data[quantity_col], 
                   color=colour[p_ind], marker='_', s=step_width, lw=8)

        # line_buffer and step_width can be altered to fit the profile.
        for rstep_ind in range(1, len(reac_path_data)):
            ax.plot([reac_path_data['Rx'].iloc[rstep_ind-1]+line_buffer, 
                     reac_path_data['Rx'].iloc[rstep_ind]-line_buffer], 
                     [reac_path_data[quantity_col].iloc[rstep_ind-1], 
                     reac_path_data[quantity_col].iloc[rstep_ind]],  
                     color=colour[p_ind], linestyle='--')

            # Plot labels with dataframe index and energy label.
            if label == True:
                step_label = reac_path_data.index.values[rstep_ind] + \
                             '\n(' + str(int(reac_path_data[
                             quantity_col].iloc[rstep_ind])) + ')'
                ax.text(reac_path_data['Rx'].iloc[rstep_ind], 
                        reac_path_data[quantity_col].iloc[rstep_ind]+6, 
                        step_label, color=colour[p_ind], fontsize=11, 
                        horizontalalignment='center')
        
        # Plot labels of reactants.
        if label == True:
            reactant_label = reac_path_data.index.values[0] + \
                             '\n(' + str(int(reac_path_data[
                             quantity_col].iloc[0])) + ')'
            ax.text(reac_path_data['Rx'].iloc[0], 
                    reac_path_data[quantity_col].iloc[0]+6, 
                    reactant_label, color=colour[p_ind], 
                    fontsize=11, horizontalalignment='center')

    # Set figure properties.
    ax.set_xlabel('R$_{x}$', fontsize=13)
    ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
    ax.set_xticks([])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

def normalise_parameters(mol_data, geom_params):
    """
    Update DataFrame with bond/angle/dihedrals mapped to a 0:1 range.

    Distances are normalised to [0:1] range
    Angles are mapped from [0:180] range to [0:1] range
    Dihedrals are mapped from [-180:180] range to [0:1] range

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results to be normalised.
    geom_params : `dict`
        Key is column heading and 
        value is the atom indexes of the parameter.

    Returns
    -------
    param_headings : `list of str`
        Parameter headings for the normalised parameters.

    """
    param_headings = []
    for key, value in geom_params.items():
        if len(value) == 2:
            mol_data["Norm " + key] = mol_data[key]/mol_data[key].max()
        elif len(value) == 3:
            mol_data["Norm " + key] = mol_data[key]/180.
        else:
            mol_data["Norm " + key] = (mol_data[key]%360.)/360.

        # Set parameter heading
        param_headings.append("Norm " + key)

    return param_headings

def set_mol_colours(mol_data, energy_col):
    """
    Set colours for different conformers, can represent energy values.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    colours : `list`
        Colour code corresponding to each conformer.
    
    """
    # Calculate normalised energy to plot colour by if given.
    if energy_col != None:
        mol_data['Norm E'] = mol_data[energy_col]/mol_data[energy_col].max()
        # colmap = sns.cubehelix_palette(start=2.5, rot=.5, dark=0, light=0.5, as_cmap=True)
        colmap = sns.cubehelix_palette(as_cmap=True)
        for val in mol_data['Norm E']:
            colours = [colmap(val)[:3] for val in mol_data['Norm E']]
    else:
    # Else set colours different for each conformer.
        colours = sns.cubehelix_palette(len(mol_data.index))
    return colours


def plot_mols_radar(mol_data, geom_params, save=None, 
                    colour=None, energy_col=None):
    """
    Plot molecules against multiple geometric parameters in a radial plot.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    geom_params : `dict`
        Key is column heading and value is the atom indexes of the parameter.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : `list of str`
        Colour list to plot conformers by.
        [default: None type; uses cubehelix colours].
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = radial_plot_setup()

    # Calculate angles to plot, set parameter list
    num_params = len(geom_params.keys())
    plot_angles = [n / float(num_params) * 2 * np.pi for n in range(num_params)]
    plot_angles += plot_angles[:1]

    # Normalise molecule parameters.
    param_headings = normalise_parameters(mol_data, geom_params)
    param_headings.append(param_headings[0])

    # Set colour for molecules.
    if colour == None:
        mol_data['Colour'] = set_mol_colours(mol_data, energy_col)
    else:
        mol_data['Colour'] = colour

    # Plot for each conformer
    for mol in mol_data.index:
        ax.plot(plot_angles, mol_data.loc[mol, param_headings], 
                label=mol, color=mol_data.loc[mol, 'Colour'])
        ax.fill(plot_angles, mol_data.loc[mol, param_headings], 
                color=mol_data.loc[mol, 'Colour'], alpha=0.1)

    # Set plot attributes
    ax.set_xticks(plot_angles[:-1])
    ax.set_xticklabels(list(geom_params.keys()))
    ax.set_yticks([])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, 
              frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def plot_mol_map(mol_data, geom_params, save=None, 
                  colour=None, energy_col=None):
    """
    Plot molecules against several geometric parameters in a linear plot.

    Parameters
    ----------
    mol_data:  :pandas:`DataFrame`
        DataFrame containing molecule results.
    geom_params : `dict`
        Key is column heading and value is the atom indexes of the parameter.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : `list of str`
        Colour list to plot conformers by.
        [default: None type; uses cubehelix colours].
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()
    num_params = len(geom_params.keys())
    plot_params = list(geom_params.keys())

    # Normalise molecule parameters.
    param_headings = normalise_parameters(mol_data, geom_params)

    # Set colour for molecules.
    if colour == None:
        mol_data['Colour'] = set_mol_colours(mol_data, energy_col)
    else:
        mol_data['Colour'] = colour

    # Plot data.
    for i, mol in enumerate(mol_data.index):
        ax.plot(range(num_params), mol_data.loc[mol, param_headings], label=mol, 
                color=mol_data.loc[mol, 'Colour'], marker='o', alpha=0.8)

    # Set x and y labels and ticks
    ax.set_xticks(range(num_params))
    ax.set_xticklabels(plot_params, rotation=20, ha='right')
    ax.set_ylim(ymin=0.0, ymax=1.0)

    # Set legend.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, 
              handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax
