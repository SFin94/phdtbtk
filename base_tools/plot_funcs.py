"""General plotting functions - mainly use Molecule DataFrames."""

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

def plot_setup(figsize_x=8, figsize_y=6, fig=None, ax=None):
    """
    Set up general plot.

    Parameters
    ----------
    figsize_x : :class:`int`
        x dimension of plot [default: 8]

    figsize_y : :class:`int`
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


def radial_plot_setup(figsizeX=6, figsizeY=6, fig=None, ax=None):
    """
    Set up radial plot.

    Parameters
    ----------
    figsize_x : :class:`int`
        x dimension of plot [default: 8]

    figsize_y : :class:`int`
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

    # Set figure and plot param(s) vs energy
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsizeX,figsizeY), 
                               subplot_kw=dict(projection='polar'))

    # ax.spines["circle"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax


def radial_plot_setup(figsize_x=6, figsize_y=6, fig=None, ax=None):
    """
    Initialise radial plot with general settings.

    Parameters
    ----------
    figsize_x : :class:`int`
        x dimension of plot [default: 6]

    figsize_y : :class:`int`
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

def plot_mols_quantity(molecule_df, 
                      quantity_column=None, 
                      save=None, 
                      colour=None, 
                      mol_labels=None, 
                      line=False,
                      fig=None, 
                      ax=None):

    """
    Plot molecules/conformers against quantity (default relative energy).

    Parameters
    ----------
    molecule_df : :pandas: `DataFrame`
        DataFrame of Molecules and properties.

    quantity_column : :class:`iterable` of :class:`str`
        Column header/s corresponding to quantity to 
        plot molecules by. If multiple then plotted on same axis.
        Single `str` can be passed for single column header.
        [Default: None] If `None` plots relative e or g. 

    save : :class:`str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]

    colour : :class:`list` of :class:`str`
        Colour code to plot each quantity by.
        [Default: `None`] If `None` uses cubehelix colours.

    mol_labels : :class:`list` of :class:`str`
        Molecule identifiers if different to DatFrame index.
        [Default: None] If `None` uses DataFrame index.

    line : :class:`bool`
        If ``True`` connects scatterpoints by lines.
    
    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: `None`]
    
    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    # Set quantity column if not set.
    if quantity_column is None:
        if 'relative g' in molecule_df.columns:
            quantity_column = ['relative g']
        else:
            quantity_column = ['relative e']

    # Handle if single value.
    if not isinstance(quantity_column, (list, tuple)):
        quantity_column = [quantity_column]

    # Set cubehelix colour for each quantity.
    if colour == None:
        colour = sns.cubehelix_palette(len(quantity_column), 
                                       start=.5, 
                                        dark=0, light=0.5)

    # Set up plot.
    fig, ax = plot_setup(fig=fig, ax=ax)

    # Plot conformers for each quantity.
    for i, quantity in enumerate(quantity_column):
        ax.scatter(list(molecule_df.index), molecule_df[quantity], 
                        marker='o', alpha=0.8, color=colour[i], 
                        label=quantity, s=70)
        if line == True:
            ax.plot(list(molecule_df.index), molecule_df[quantity], alpha=0.3, 
                         color=colour[i], ls='--')

    if mol_labels == None:
        mol_labels = molecule_df.index

    # Set x and y labels and ticks
    ax.set_xticklabels(mol_labels, rotation=15)
    if len(quantity_column) == 1:
        y_label = quantity_column[0]
    else:
        y_label = 'relative quantity'
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel('Molecule', fontsize=13)
    plt.legend(fontsize=13)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax

def plot_param_quantity(molecule_df,
                        parameter_column,
                        quantity_column=None, 
                        save=None, 
                        colour=None, 
                        mol_labels=None, 
                        line=False,
                        fig=None, 
                        ax=None):
    """
    Plot molecules/conformers against the relative energy.

    Parameters
    ----------
    molecule_df : :pandas:`DataFrame`
        DataFrame of Molecules and properties.

    parameter_column : :class:`str`
        Column header of parameter to plot.

    quantity_column : :class:`str`
        Column header of quantity to plot.
        [Default: None] If `None` plots relative e or g.
    
    save : :class:`str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]

    colour : :class:`list` of :class:`str`
        Colour codes to plot molecules by.
        [Default: `None`] If `None` uses cubehelix colours.

    mol_labels : :class:`list` of :class:`str`
        Molecule identifiers if different to DatFrame index.
        [Default: None] If `None` uses DataFrame index.

    line : :class:`bool`
        If ``True`` connects scatterpoints by lines.

    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: `None`]

    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    # Set quantity column if not set.
    if quantity_column is None:
        if 'relative g' in molecule_df.columns:
            quantity_column = 'relative g'
        else:
            quantity_column = 'relative e'

    # Set up plot.
    fig, ax = plot_setup(fig=fig, ax=ax)

    # Set cubehelix colour for each conformer or if opt or not.
    if colour == None:
        if 'opt' in molecule_df.columns.values:
            colours_opt = ['#F26157', '#253237']
            colour = [colours_opt[opt] for opt in molecule_df['opt']]

            # Set legend 
            ax.legend(handles=[mlin.Line2D([], [], color=colours_opt[0], 
                               label='Unoptimised', marker='o', alpha=0.6, 
                               linestyle=' '), 
                               mlin.Line2D([], [], color=colours_opt[1], 
                               label='Optimised', marker='o', alpha=0.6, 
                               linestyle=' ')], 
                               frameon=False, handletextpad=0.1, fontsize=10)
        else:
            colour = sns.cubehelix_palette(len(molecule_df),
                                       start=.5, rot=-0.4,
                                       dark=0, light=0.5)

    # Plot points and connecting lines.
    ax.scatter(molecule_df[parameter_column], molecule_df[quantity_column], 
               color=colour, marker='o', s=70, alpha=0.8)
    if line == True:
        ax.plot(molecule_df.sort_values(parameter_column)[parameter_column], 
                molecule_df.sort_values(parameter_column)[quantity_column], 
                marker=None, alpha=0.4, color=colour[1])

    # Plot settings.
    ax.set_ylabel(f'$\Delta$ {quantity_column[10:]}', fontsize=13)
    ax.set_xlabel(parameter_column, fontsize=13)
    # plt.legend(fontsize=13)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax

def plot_PES(molecule_df, 
             parameter_columns, 
             quantity_column=None,
             save=None, 
             colour=None, 
             opt_filter=True,
             fig=None, 
             ax=None):
    """
    Plot 2D PES for two parameters.

    Parameters
    ----------
    molecule_df : :pandas:`DataFrame`
        DataFrame of Molecules and properties.

    parameter_columns : :class:`iterable` of :class:`str`
        Column headers for two parameters to plot.

    quantity_column : :class:`str`
        Column header of quantity to plot.
        [Default: None] If `None` plots relative e or g.
    
    save : :class:`str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]

    colour : :class:`list` of :class:`str`
        Colour codes to plot molecules by.
        [Default: `None`] If `None` uses cubehelix colours.

    opt_filter : :class:`bool`
        If ``True`` removes unoptimised points from interpolation.

    fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
        [default: `None`]

    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    # Set quantity column if not set.
    if quantity_column is None:
        if 'relative g' in molecule_df.columns:
            quantity_column = 'relative g'
        else:
            quantity_column = 'relative e'

    # Set up plot.
    fig, ax = plot_setup(fig=fig, ax=ax)

    # Remove unoptimised points.
    if opt_filter is True:
        molecule_df = molecule_df[molecule_df.opt]

    # Set linearly spaced parameter values and define grid.
    param_one_range = np.linspace(molecule_df[parameter_columns[0]].min(), 
                                  molecule_df[parameter_columns[0]].max(), 
                                  100)
    param_two_range = np.linspace(molecule_df[parameter_columns[1]].min(), 
                                  molecule_df[parameter_columns[1]].max(), 
                                  100)
    param_one_grid, param_two_grid = np.meshgrid(param_one_range, param_two_range)

    # Interpolate quantity value on to the grid points.
    interp_quant = griddata((molecule_df[parameter_columns[0]].values, 
                             molecule_df[parameter_columns[1]].values), 
                             molecule_df[quantity_column], 
                             (param_one_grid, param_two_grid))

    # Set cmap if none provided.
    if colour == None:
        colour = sns.cubehelix_palette(dark=0, as_cmap=True)

    # Plot filled contour and add colour bar.
    c = ax.contourf(param_one_range, param_two_range, 
                    interp_quant, 20, cmap=colour, vmax=150)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label(f'$\Delta$ {quantity_column[9:]}', fontsize=13)

    # Set x and y labels
    ax.set_xlabel(parameter_columns[0], fontsize=13)
    ax.set_ylabel(parameter_columns[1], fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


# def plot_reaction_profile(reaction_data, quantity_column='Relative G', save=None, colour=None, step_width=3000, line_buffer=0.08, label=True):

#     '''Function which plots a reaction profile

#     Parameters:
#      reaction_data: pandas DataFrame
#      energy_col: str - header for relative energy column in dataframe [default: 'Relative E']
#      save: str - name of image to save plot too (minus .png extension) [deafult: None type]
#      colour: matplotlib cmap colour - colour map to generate path plot colours from [default: None type; if default then a cubehelix colour map is used].
#      step_width: int - the marker size of the scatter hlines used to mark the reaction steps [default: 3000]
#      line_buffer: float - the buffer from the centre of the hline that the connecting lines will connect from [default: 0.05]
#      label: bool - if True then plots the indexes with each step, if False then returns the figure without labels

#     Returns:
#      fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

#     '''

#     fig, ax = plot_setup()
#     paths = list(reaction_data['Reaction path'].unique())
#     label_buffer = line_buffer - 0.01

#     # Set colours if not provided - the number of paths will be number of colours
# #    colours = sns.cubehelix_palette(len(paths))
#     if colour == None:
#         col_pallete = sns.color_palette("cubehelix", len(paths))
#         colour = []
#         for p_ind in range(len(paths)):
#             colour.append(col_pallete[paths.index(p_ind)])

#     # Plot the lines and points for the profile (line_buffer and step_width can be altered to fit the profile)
#     for p_ind, path in enumerate(paths):
#         reac_path_data = reaction_data.loc[reaction_data['Reaction path'] == path]
#         ax.scatter(reac_path_data['Reaction coordinate'], reac_path_data[quantity_column], color=colour[p_ind], marker='_', s=step_width, lw=5)
#         for rstep_ind in range(1, len(reac_path_data)):
#             ax.plot([reac_path_data['Reaction coordinate'].iloc[rstep_ind-1]+line_buffer, reac_path_data['Reaction coordinate'].iloc[rstep_ind]-line_buffer], [reac_path_data[quantity_column].iloc[rstep_ind-1], reac_path_data[quantity_column].iloc[rstep_ind]],  color=colour[p_ind], linestyle='--')

#             # Plot labels with dataframe index and energy label unless False, plot reactants at the end
#             if label == True:
#                 step_label = reac_path_data.index.values[rstep_ind] + ' (' + str(int(reac_path_data[quantity_column].iloc[rstep_ind])) + ')'
#                 ax.text(reac_path_data['Reaction coordinate'].iloc[rstep_ind]-label_buffer, reac_path_data[quantity_column].iloc[rstep_ind]+6, step_label, color=colour[p_ind], fontsize=11)

#         if label == True:
#             reactant_label = reac_path_data.index.values[0] + ' (' + str(int(reac_path_data[quantity_column].iloc[0])) + ')'
#             ax.text(reac_path_data['Reaction coordinate'].iloc[0]-label_buffer, reac_path_data[quantity_column].iloc[0]+6, reactant_label, color=colour[p_ind], fontsize=11)

#     # Set x and y labels
#     ax.set_xlabel('R$_{x}$', fontsize=13)
#     ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
#     ax.set_xticks([])

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax


# def normalise_parameters(conformer_data, geom_parameters):
#     """
#     Normalise geometric (bond/angle/dihedral) parameter values (all scaled to 0:1 range).

#     Distances are normalised to [0:1] range.
#     Angles are mapped from [0:180] range to [0:1] range.
#     Dihedrals are mapped from [-180:180] range to [0:1] range.

#     Updates DataFrame in place.

#     Parameters
#     ----------
#     conformer_data: pandas `DataFrame`
#         Conformer data containing values for geometric parameters to plot.
#     geom_parameters: `list`
#         Column headings defining the parameters.
    
#     Returns
#     -------
#     param_headings: `list` of `str`
#         parameter headings for the normalised parameters
    
#     """
#     param_headings = []
#     for parameter in geom_parameters:
#         if len(value) == 2:
#             conformer_data["Norm " + key] = conformer_data[parameter]/conformer_data[parameter].max()
#         elif len(value) == 3:
#             conformer_data["Norm " + parameter] = conformer_data[parameter]/180.
#         else:
#             conformer_data["Norm " + parameter] = (conformer_data[parameter]%360.)/360.
        
#         # Set parameter heading
#         param_headings.append("Norm " + parameter)

#     return param_headings


# def set_conformer_colours(conformer_data, energy_col):

#     '''Function that sets the colour for different conformers which can be normalised by energy values

#     Parameters:
#      conformer_data: pandas DataFrame - conformer data
#      energy_col: str - name of the dataframe column header corresponding to the thermodynamic quantity to normalise the colours of the conformers too

#     Returns:
#      colblock/col_vals: list - colour code corresponding to each conformer 
#     '''

#     # Calculate normalised energy to plot colour by if given
#     if energy_col != None:
#         conformer_data['Norm E'] = conformer_data[energy_col]/conformer_data[energy_col].max()
#         # colmap = sns.cubehelix_palette(start=2.5, rot=.5, dark=0, light=0.5, as_cmap=True)
#         colmap = sns.cubehelix_palette(as_cmap=True)
#         for val in conformer_data['Norm E']:
#             colour_vals = [colmap(val)[:3] for val in conformer_data['Norm E']]    
#         return colour_vals
#     else:
#     # Else set colours different for each conformer 
#         colblock = sns.cubehelix_palette(len(conformer_data.index))
#         return colblock

    
# def plot_conf_radar(conformer_data, geom_parameters, save=None, colour=None, energy_col=None):
#     """
#     Plot conformers against several geometric parameters in a radial plot.

#     Parameters
#     ----------
#     conformer_data: pandas `DataFrame`
#         Conformer data containing values for geometric parameters to plot.
#     geom_parameters: `list`
#         Column headings defining the parameters.
#     save: `str`
#         Name of image to save plot too (minus .png extension) [deafult: None type].
#     colour: :matplotlib:`cmap`
#         Colour map to generate path plot colours from 
#         [default: None type; if default then a cubehelix colour map is used].
#     energy_col: `str`
#         Column heading of optional energy parameter to colour rank conformers by.
#         [default: None type]

#     Returns:
#      fig, ax - :matplotlib:`fig`, :matplotlib:`ax` objects for the plot

#     """
#     fig, ax = radial_plot_setup()

#     # Calculate angles to plot, set parameter list
#     num_params = len(geom_parameters)
#     plot_angles = [n / float(num_params) * 2 * np.pi for n in range(num_params)]
#     plot_angles += plot_angles[:1]

#     # Normalise conformer parameters
#     param_headings = normalise_parameters(conformer_data, geom_parameters)
#     param_headings.append(param_headings[0])
    
#     # Set colour
#     if colour == None:
#         conformer_data['Colour'] = set_conformer_colours(conformer_data, energy_col)
#     else:
#         conformer_data['Colour'] = colour

#     # Plot for each conformer
#     for conf in conformer_data.index:
#         ax.plot(plot_angles, conformer_data.loc[conf][param_headings], label=conf, color=conformer_data.loc[conf]['Colour'])
#         ax.fill(plot_angles, conformer_data.loc[conf][param_headings], color=conformer_data.loc[conf]['Colour'], alpha=0.1)

#     # Set plot attributes
#     ax.set_xticks(plot_angles[:-1])
#     ax.set_xticklabels(geom_parameters)
#     ax.set_yticks([])
#     ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, handletextpad=0.1, fontsize=9)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax


# def plot_conf_map(conformer_data, geom_parameters, save=None, colour=None, energy_col=None):
#     """
#     Plot conformers against several geometric parameters in a linear map.

#     Parameters
#     ----------
#     conformer_data: pandas `DataFrame`
#         Conformer data containing values for geometric parameters to plot.
#     geom_parameters: `list`
#         Column headings defining the parameters.
#     save: `str`
#         Name of image to save plot too (minus .png extension) [deafult: None type].
#     colour: :matplotlib:`cmap`
#         Colour map to generate path plot colours from 
#         [default: None type; if default then a cubehelix colour map is used].
#     energy_col: `str`
#         Column heading of optional energy parameter to colour rank conformers by.
#         [default: None type]

#     Returns:
#      fig, ax - :matplotlib:`fig`, :matplotlib:`ax` objects for the plot

#     """
#     fig, ax = plot_setup()
#     num_params = len(geom_parameters)

#     # Normalise conformer parameters
#     param_headings = normalise_parameters(conformer_data, geom_parameters)

#     # Set colour
#     if colour == None:
#         conformer_data['Colour'] = set_conformer_colours(conformer_data, energy_col)
#     else:
#         conformer_data['Colour'] = colour

#     # Plot data
#     for cInd, conf in enumerate(conformer_data.index):
#         ax.plot(range(num_params), conformer_data.loc[conf][param_headings], label=conf, color=conformer_data.loc[conf]['Colour'], marker='o', alpha=0.8)

#     # Set x and y labels and ticks
#     ax.set_xticks(range(num_params))
#     ax.set_xticklabels(plot_params, rotation=20, ha='right')
#     ax.set_ylim(ymin=0.0, ymax=1.0)

#     ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, handletextpad=0.1, fontsize=9)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
    
#     # if energy_col != None:
#     #     ax_cbar = inset_axes(ax, width="50%", height="3%", loc='upper right')
#     #     plt.colorbar(cm.ScalarMappable(cmap=colmap), ax=ax, cax=ax_cbar, orientation="horizontal", ticks=[0, 1], label='$\Delta$G')

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax


# def plot_order(molecule_df_one, quantity_column_one, molecule_df_two, quantity_column_two, save=None, fig=None, ax=None):
#     """
#     Plot comparative scatter plot of rankings for two sets of data.
    
#     Parameters
#     ----------
#     molecule_df_one: pandas `DataFrame`
#         Conformer data containing quantity one values.
#     quantity_column_one: `str`
#         Column heading of quantity one to rank conformers by.
#     molecule_df_two: pandas `DataFrame`
#         Conformer data (same conformers as molecule_df_one) containing quantity two values.
#     quantity_column_two: `str`
#         Column heading of quantity two to rank conformers by.
#     save: `str`
#         Name of image to save plot too (minus .png extension) [deafult: None type].
#     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
#         [default: NoneType]
    
#     Returns
#     -------
#      fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

#     """
#     fig, ax = plot_setup(figsizeX=6, figsizeY=6, fig=fig, ax=ax)
    
#     # Set order of molecules by first quantity
#     mol_order = list((molecule_df_one.sort_values(quantity_column_one).index))

#     # Get rank of columns and compare - sorts by lowest first
#     rank = []
#     for molecule_df, col in zip([molecule_df_one, molecule_df_two], [quantity_column_one, quantity_column_two]):
#         order = list((molecule_df.sort_values(col).index))
#         rank.append([mol_order.index(i) for i in order])
    
#     # Plot scatter of ranks
#     ax.scatter(rank[0], rank[1], color='#5C6B73', s=150)

#     # Set x and y axis as integer numbers
#     conf_indexes = [int(i) for i in range(len(rank[0]))]
#     ax.set_xticks(range(len(conf_indexes)))
#     ax.set_yticks(range(len(conf_indexes)))
#     ax.set_xticklabels(conf_indexes, fontsize=10)
#     ax.set_yticklabels(conf_indexes, fontsize=10)

#     if save != None:
#         plt.savefig(save + '.png')
    
#     return fig, ax


# def plot_conf_comparison(molecule_df_one, quantity_column_one, molecule_df_two, quantity_column_two, labels, colours=['#9AD0BB','#5C6B73'], save=None):
#     """
#     Plot bar chart of relative quantities for two sets of conformers.
    
#     Parameters
#     ----------
#     molecule_df_one: pandas `DataFrame`
#         Conformer data containing quantity one values.
#     quantity_column_one: `str`
#         Column heading of quantity one to plot.
#     molecule_df_two: pandas `DataFrame`
#         Conformer data (same conformers as molecule_df_one) containing quantity two values.
#     quantity_column_two: `str`
#         Column heading of quantity to plot.
#     labels: `list of str`
#         Labels for each of the conformer sets.
#     save: `str`
#         Name of image to save plot too (minus .png extension) [deafult: None type].
#     fig, ax: :matplotlib:fig, :matplotlib:ax objects for the plot
#         [default: NoneType]
    
#     Returns
#     -------
#      fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

#     """
#     # Initialise plot and variables.
#     fig, ax = plot_setup()
#     width=0.4
#     x_range = np.arange(len(molecule_df_one.index))

#     # Plot bar chart for each conformer data set.
#     ax.bar(x_range - width/2, molecule_df_one[quantity_column_one], width, color=colours[0], label=labels[0], alpha=0.9)
#     ax.bar(x_range + width/2, molecule_df_two[quantity_column_two], width, color=colours[1], label=labels[1], alpha=0.9)

#     # Set plot properties.
#     ax.set_xticks(x_range)
#     ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)')
#     ax.set_xlabel('Conformer')
#     plt.legend()

#     if save != None:
#         plt.savefig(save + '.png')
    
#     return fig, ax



# ### Plot Functions from molLego - 3rd Feb 2021

# """Module containing general plotting routines for molecules."""

# import sys
# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.axes as axes
# import matplotlib.lines as mlin
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from scipy.interpolate import griddata

# def plot_setup(figsize_x=8, figsize_y=6, fig=None, ax=None):
#     """
#     Initialise plot with general settings.

#     Parameters
#     ----------
#     figsize_x : `int`
#         x dimension of plot [default: 8]
#     figsize_y : `int`
#         y dimension of plot [default: 6]
#     fig : :matplotlib:`fig`
#         If other type of plot is called first [default: None]
#     ax : :matplotlib:`axes`
#         If other type of plot is called first [default: None]

#     Returns
#     -------
#     fig, ax: :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
#     """
#     # Set font parameters and colours
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = 'Arial'
#     colour_grey = '#3E3E3E'
#     plt.rcParams.update({'text.color': colour_grey, 
#                          'axes.labelcolor': colour_grey, 
#                          'xtick.color': colour_grey, 
#                          'ytick.color': colour_grey})

#     # Initiaise figure.
#     if fig == None and ax == None:
#         fig, ax = plt.subplots(figsize=(figsize_x,figsize_y))

#     # Remove lines from plot frame.
#     ax.spines["top"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.tick_params(labelsize=12)

#     return fig, ax


# def radial_plot_setup(figsize_x=6, figsize_y=6, fig=None, ax=None):
#     """
#     Initialise radial plot with general settings.

#     Parameters
#     ----------
#     figsize_x : `int`
#         x dimension of plot [default: 6]
#     figsize_y : `int`
#         y dimension of plot [default: 6]
#     fig : :matplotlib:`fig`
#         If other type of plot is called first [default: None]
#     ax : :matplotlib:`axes`
#         If other type of plot is called first [default: None]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
#     """
#     # Set font parameters and colours
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = 'Arial'
#     colour_grey = '#3E3E3E'
#     plt.rcParams.update({'text.color': colour_grey, 
#                          'axes.labelcolor': colour_grey, 
#                          'xtick.color': colour_grey, 
#                          'ytick.color': colour_grey})

#     # Set figure and plot param(s) vs energy
#     if fig == None and ax == None:
#         fig, ax = plt.subplots(figsize=(figsize_x,figsize_y), 
#                                subplot_kw=dict(projection='polar'))

#     # ax.spines["circle"].set_visible(False)
#     ax.tick_params(labelsize=12)

#     return fig, ax

# def plot_mols_E(molecule_df, energy_col=['Relative E'], save=None, 
#                 colour=None, mol_labels=None, line=False):
#     """
#     Plot molecules/conformers against relative energy.

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     energy_col : str`
#         Column header corresponding to quantity to plot molecules by.
#         [default: Relative E]
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [default: None; no figure is saved]
#     colour : `list of str`
#         Colour list to plot molecules by.
#         [default: None type; uses cubehelix colours].
#     mol_labels : `list of str`
#         Molecule identifiers if different to DatFrame index.
#         [default: None type, uses DataFrame index]
#     line : `bool`
#         If True, connects scatterpoints by lines.
        
#     Returns
#     -------
#     fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

#     """
#     fig, ax = plot_setup(figsize_x=8, figsize_y=7)

#     # Plot conformer vs. relative energy
#     if type(energy_col) != list:
#         energy_col = [energy_col]
#     if colour == None:
#         colour = sns.cubehelix_palette(len(energy_col), start=.5, 
#                                      rot=-.4, dark=0, light=0.5)

#     # Plot conformers for each quantity
#     for col_ind, col in enumerate(energy_col):
#         ax.scatter(list(molecule_df.index), molecule_df[col], 
#                         marker='o', alpha=0.8, color=[col_ind], 
#                         label=col, s=70)
#         if line == True:
#             ax.plot(list(molecule_df.index), molecule_df[col], alpha=0.3, 
#                          color=colour[col_ind], ls='--')

#     if mol_labels == None:
#         mol_labels = molecule_df.index

#     # Set x and y labels and ticks
#     ax.set_xticklabels(mol_labels, rotation=15)
#     ax.set_ylabel('Relative Energy (kJmol$^{-1}$)', fontsize=13)
#     ax.set_xlabel('Molecule', fontsize=13)
#     plt.legend(fontsize=13)

#     if save != None:
#         plt.savefig(save + '.png', dpi=600)

#     return fig, ax


# def plot_mols_thermo(molecule_df, save=None, mol_labels=None, enthalpy=False):
#     """
#     Plot molecules against relative E and G (and H).

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure is saved]
#     mol_labels : `list of str`
#         Molecule identifiers if different to DatFrame index.
#         [default: None type, uses DataFrame index]
#     enthalpy :
#         If True also plots enthalpy values.
#         [default: False]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = plot_setup()

#     # Set colours for G and E
#     e_colour = '#245F6B'
#     g_colour = '#D17968'

#     # Plot products vs. energy
#     ax.scatter(molecule_df.index, molecule_df['Relative E'], 
#                color=e_colour, s=70, label='$\Delta$E')
#     ax.scatter(molecule_df.index, molecule_df['Relative G'], 
#                color=g_colour, s=70, label='$\Delta$G')

#     # Plot enthalpy (H) too if flag is True
#     if enthalpy == True:
#         h_colour = '#175443'
#         ax.scatter(molecule_df.index, molecule_df['Relative H'], 
#                    color=h_colour, s=70, label='$\Delta$H')

#     # Set labels and axis settings
#     if labels == None:
#         labels = list(molecule_df.index)

#     ax.tick_params(labelsize=10)
#     ax.set_xticklabels(labels, rotation=15, fontsize=11)
#     ax.set_ylabel('Relative E/G (kJmol$^{-1}$)', fontsize=11)
#     ax.set_xlabel('Molecule', fontsize=11)
#     ax.legend(frameon=False, loc=1)

#     if save != None:
#         plt.savefig(save + '.png', dpi=600)

#     return fig, ax


# def plot_param_E(molecule_df, param_col, energy_col='Relative E', 
#                  save=None, colour=None, scan=False):
#     """
#     Plot relative energies (or other specified quantity) of molecules.

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     param_col : `str`
#         Column header corresponding to geometric parameter to plot.
#     energy_col : `str`
#         Column header corresponding to quantity to plot molecules by.
#         [default: Relative E]
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure saved]
#      colour : `str`
#         A colour specified by a recognised matplotlib format.
#         [default: None type; uses cubehelix colours].
#      scan : `bool`
#         If True, connects scatterpoints by lines.
#         [default: False]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = plot_setup()

#     # Set colours for plotting if not provided
#     if colour == None:
#         colour = ['#D17968', '#12304e']
#     elif len(colour[0]) == 1:
#         colour = [colour]

#     # Set colours by molecule opt value.
#     if 'Optimised' in molecule_df.columns.values:
#         colour_list = []
#         [colour_list.append(colour[opt]) for opt in molecule_df['Optimised']]
#     elif len(colour) == len(list(molecule_df.index)):
#         colour_list = colour
#     else:
#         colour_list = [colour[0]]*len(list(molecule_df.index))

#     # Plot points and connecting lines if scan
#     ax.scatter(molecule_df[param_col], molecule_df[energy_col], 
#                color=colour_list, marker='o', s=70, alpha=0.8)
#     if scan == True:
#         ax.plot(molecule_df[param_col], molecule_df[energy_col], 
#                 marker=None, alpha=0.4, color=colour[1])

#     # Set x and y labels
#     ax.set_xlabel(param_col, fontsize=11)
#     ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=11)

#     # Set legend to show unopt vs. opt points
#     if 'Optimised' in molecule_df.columns.values:
#         ax.legend(handles=[mlin.Line2D([], [], color=colour[0], 
#                   label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), 
#                   mlin.Line2D([], [], color=colour[1], label='Optimised', 
#                   marker='o', alpha=0.6, linestyle=' ')], frameon=False, 
#                   handletextpad=0.1, fontsize=10)

#     if save != None:
#         plt.savefig(save + '.png', dpi=600)

#     return fig, ax


# def plot_PES(molecule_df, param_cols, energy_col='Relative E', 
#              save=None, colour=None, opt_filter=True):
#     """
#     Plot 2D PES for two geometric parameters.

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     param_cols : `list of str`
#         Column headers corresponding to the two geometric parameter to plot.
#     energy_col : ``str``
#         Column header corresponding to quantity to plot molecules by.
#         [default: Relative E]
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure is saved]
#     colour : :matplotlib:`cmap`
#         Colour map to plot PES. 
#         [default: None type; uses cubehelix colour map].
#     opt_filter : `bool`
#         If True then removes unoptimised data points.
#         [default: True]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = plot_setup(figsize_x=7.5, figsize_y=6)

#     # Filter out any unoptimised points if optimised present
#     opt_col = ('Optimised' in molecule_df.columns.values)
#     if all([opt_filter, opt_col]):
#         molecule_df = molecule_df[molecule_df.Optimised]

#     # Set linearly spaced parameter values and define grid between them
#     param_one_range = np.linspace(molecule_df[param_cols[0]].min(), 
#                                   molecule_df[param_cols[0]].max(), 100)
#     param_two_range = np.linspace(molecule_df[param_cols[1]].min(), 
#                                   molecule_df[param_cols[1]].max(), 100)
#     param_one_grid, param_two_grid = np.meshgrid(param_one_range, 
#                                                  param_two_range)

#     # Interpolate the energy data on to the grid points for plotting
#     interp_E = griddata((molecule_df[param_cols[0]].values, 
#                          molecule_df[param_cols[1]].values), 
#                          molecule_df[energy_col], 
#                          (param_one_grid, param_two_grid))

#     # Set cmap if none provided
#     if colour == None:
#         colour = sns.cubehelix_palette(dark=0, as_cmap=True)

#     # Plot filled contour and add colour bar
#     c = ax.contourf(param_one_range, param_two_range, interp_E, 
#                     20, cmap=colour, vmax=150)
#     fig.subplots_adjust(right=0.8)
#     cb = fig.colorbar(c)
#     cb.set_label('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

#     # Set x and y labels
#     ax.set_xlabel(param_cols[0], fontsize=13)
#     ax.set_ylabel(param_cols[1], fontsize=13)

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax

# def plot_reaction_profile(reaction_data, quantity_column='Relative G', save=None,
#                             colour=None, step_width=3000, line_buffer=0.08, 
#                             label=True, fig=None, ax=None):
#     """
#     Plot a reaction profile.

#     Parameters
#     ----------
#     reaction_data : :pandas:`DataFrame`
#         The reaction profile dataframe to plot.
#     energy_col : ``str``
#         Column header of quantity to plot reaction steps by.
#         [default: 'Relative G']
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure is saved]
#     colour : :matplotlib:`cmap`
#         Colour map to generate path plot colours from. 
#         [default: None type; uses cubehelix colour map].
#     step_width : `int`
#         The marker size of the scatter hlines used to mark the reaction step.
#         [default: 3000]
#     line_buffer : `float`
#         The buffer from centre of the hline of position
#         the connecting lines will connect to.
#         [default: 0.05]
#     label : `bool`
#         If True then plots the indexes with each step.
#         If False then returns the figure without labels.
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = plot_setup(fig=fig, ax=ax)
#     paths = list(reaction_data['Reaction path'].unique())

#     # Set colours if not provided.
#     if colour == None:
#         col_pallete = sns.color_palette("cubehelix", len(paths))
#         colour = []
#         for p_ind in range(len(paths)):
#             colour.append(col_pallete[paths.index(p_ind)])

#     # Plot lines and points for the profile.
#     for p_ind, path in enumerate(paths):
#         reac_path_data = reaction_data.loc[
#                             reaction_data['Reaction path'] == path]
#         ax.scatter(reac_path_data['Rx'], reac_path_data[quantity_column], 
#                    color=colour[p_ind], marker='_', s=step_width, lw=8)

#         # line_buffer and step_width can be altered to fit the profile.
#         for rstep_ind in range(1, len(reac_path_data)):
#             ax.plot([reac_path_data['Rx'].iloc[rstep_ind-1]+line_buffer, 
#                      reac_path_data['Rx'].iloc[rstep_ind]-line_buffer], 
#                      [reac_path_data[quantity_column].iloc[rstep_ind-1], 
#                      reac_path_data[quantity_column].iloc[rstep_ind]],  
#                      color=colour[p_ind], linestyle='--')

#             # Plot labels with dataframe index and energy label.
#             if label == True:
#                 step_label = reac_path_data.index.values[rstep_ind] + \
#                              '\n(' + str(int(reac_path_data[
#                              quantity_column].iloc[rstep_ind])) + ')'
#                 ax.text(reac_path_data['Rx'].iloc[rstep_ind], 
#                         reac_path_data[quantity_column].iloc[rstep_ind]+6, 
#                         step_label, color=colour[p_ind], fontsize=11, 
#                         horizontalalignment='center')
        
#         # Plot labels of reactants.
#         if label == True:
#             reactant_label = reac_path_data.index.values[0] + \
#                              '\n(' + str(int(reac_path_data[
#                              quantity_column].iloc[0])) + ')'
#             ax.text(reac_path_data['Rx'].iloc[0], 
#                     reac_path_data[quantity_column].iloc[0]+6, 
#                     reactant_label, color=colour[p_ind], 
#                     fontsize=11, horizontalalignment='center')

#     # Set figure properties.
#     ax.set_xlabel('R$_{x}$', fontsize=13)
#     ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
#     ax.set_xticks([])

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax

# def normalise_parameters(molecule_df, geom_params):
#     """
#     Update DataFrame with bond/angle/dihedrals mapped to a 0:1 range.

#     Distances are normalised to [0:1] range
#     Angles are mapped from [0:180] range to [0:1] range
#     Dihedrals are mapped from [-180:180] range to [0:1] range

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results to be normalised.
#     geom_params : `dict`
#         Key is column heading and 
#         value is the atom indexes of the parameter.

#     Returns
#     -------
#     param_headings : `list of str`
#         Parameter headings for the normalised parameters.

#     """
#     param_headings = []
#     for key, value in geom_params.items():
#         if len(value) == 2:
#             molecule_df["Norm " + key] = molecule_df[key]/molecule_df[key].max()
#         elif len(value) == 3:
#             molecule_df["Norm " + key] = molecule_df[key]/180.
#         else:
#             molecule_df["Norm " + key] = (molecule_df[key]%360.)/360.

#         # Set parameter heading
#         param_headings.append("Norm " + key)

#     return param_headings

# def set_mol_colours(molecule_df, energy_col):
#     """
#     Set colours for different conformers, can represent energy values.

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     energy_col : ``str``
#         Column header corresponding to quantity to code colours by.
#         [default: None]

#     Returns
#     -------
#     colours : `list`
#         Colour code corresponding to each conformer.
    
#     """
#     # Calculate normalised energy to plot colour by if given.
#     if energy_col != None:
#         molecule_df['Norm E'] = molecule_df[energy_col]/molecule_df[energy_col].max()
#         # colmap = sns.cubehelix_palette(start=2.5, rot=.5, dark=0, light=0.5, as_cmap=True)
#         colmap = sns.cubehelix_palette(as_cmap=True)
#         for val in molecule_df['Norm E']:
#             colours = [colmap(val)[:3] for val in molecule_df['Norm E']]
#     else:
#     # Else set colours different for each conformer.
#         colours = sns.cubehelix_palette(len(molecule_df.index))
#     return colours


# def plot_mols_radar(molecule_df, geom_params, save=None, 
#                     colour=None, energy_col=None):
#     """
#     Plot molecules against multiple geometric parameters in a radial plot.

#     Parameters
#     ----------
#     molecule_df : :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     geom_params : `dict`
#         Key is column heading and value is the atom indexes of the parameter.
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure is saved]
#     colour : `list of str`
#         Colour list to plot conformers by.
#         [default: None type; uses cubehelix colours].
#     energy_col : ``str``
#         Column header corresponding to quantity to code colours by.
#         [default: None]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = radial_plot_setup()

#     # Calculate angles to plot, set parameter list
#     num_params = len(geom_params.keys())
#     plot_angles = [n / float(num_params) * 2 * np.pi for n in range(num_params)]
#     plot_angles += plot_angles[:1]

#     # Normalise molecule parameters.
#     param_headings = normalise_parameters(molecule_df, geom_params)
#     param_headings.append(param_headings[0])

#     # Set colour for molecules.
#     if colour == None:
#         molecule_df['Colour'] = set_mol_colours(molecule_df, energy_col)
#     else:
#         molecule_df['Colour'] = colour

#     # Plot for each conformer
#     for mol in molecule_df.index:
#         ax.plot(plot_angles, molecule_df.loc[mol, param_headings], 
#                 label=mol, color=molecule_df.loc[mol, 'Colour'])
#         ax.fill(plot_angles, molecule_df.loc[mol, param_headings], 
#                 color=molecule_df.loc[mol, 'Colour'], alpha=0.1)

#     # Set plot attributes
#     ax.set_xticks(plot_angles[:-1])
#     ax.set_xticklabels(list(geom_params.keys()))
#     ax.set_yticks([])
#     ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, 
#               frameon=False, handletextpad=0.1, fontsize=9)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax


# def plot_mol_map(molecule_df, geom_params, save=None, 
#                   colour=None, energy_col=None):
#     """
#     Plot molecules against several geometric parameters in a linear plot.

#     Parameters
#     ----------
#     molecule_df:  :pandas:`DataFrame`
#         DataFrame containing molecule results.
#     geom_params : `dict`
#         Key is column heading and value is the atom indexes of the parameter.
#     save : `str`
#         File name to save figure as (minus .png extension). 
#         [deafult: None; no figure is saved]
#     colour : `list of str`
#         Colour list to plot conformers by.
#         [default: None type; uses cubehelix colours].
#     energy_col : ``str``
#         Column header corresponding to quantity to code colours by.
#         [default: None]

#     Returns
#     -------
#     fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

#     """
#     fig, ax = plot_setup()
#     num_params = len(geom_params.keys())
#     plot_params = list(geom_params.keys())

#     # Normalise molecule parameters.
#     param_headings = normalise_parameters(molecule_df, geom_params)

#     # Set colour for molecules.
#     if colour == None:
#         molecule_df['Colour'] = set_mol_colours(molecule_df, energy_col)
#     else:
#         molecule_df['Colour'] = colour

#     # Plot data.
#     for i, mol in enumerate(molecule_df.index):
#         ax.plot(range(num_params), molecule_df.loc[mol, param_headings], label=mol, 
#                 color=molecule_df.loc[mol, 'Colour'], marker='o', alpha=0.8)

#     # Set x and y labels and ticks
#     ax.set_xticks(range(num_params))
#     ax.set_xticklabels(plot_params, rotation=20, ha='right')
#     ax.set_ylim(ymin=0.0, ymax=1.0)

#     # Set legend.
#     ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, frameon=False, 
#               handletextpad=0.1, fontsize=9)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])

#     if save != None:
#         plt.savefig(save + '.png')

#     return fig, ax
