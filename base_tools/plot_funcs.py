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
    # ax.grid(False)
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
    ax.set_xticklabels(mol_labels, rotation=25, fontsize=7)
    if len(quantity_column) == 1:
        y_label = quantity_column[0]
    else:
        y_label = 'relative quantity'
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel('Molecule', fontsize=13)
    plt.legend(fontsize=13, frameon=False)

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
    ax.set_ylabel(f'$\Delta$ {quantity_column[9:]}', fontsize=13)
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
             max_val=None,
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
    if colour is None:
        colour = sns.cubehelix_palette(dark=0, as_cmap=True)

    # Plot filled contour and add colour bar.
    if max_val is None:
        max_val = max(molecule_df[quantity_column]+10)
    c = ax.contourf(param_one_range, param_two_range, 
                    interp_quant, 20, cmap=colour, vmax=max_val)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label(f'$\Delta$ {quantity_column[9:]}', fontsize=13)

    # Set x and y labels
    ax.set_xlabel(parameter_columns[0], fontsize=13)
    ax.set_ylabel(parameter_columns[1], fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

def normalise_parameters(molecule_df, 
                         parameter_columns):
    """
    Normalise geometric (bond/angle/dihedral) parameters to 0:1 range.

    Distances are normalised to [0:1] range.
    Angles are mapped from [0:180] range to [0:1] range.
    Dihedrals are mapped from (-180:180] range to [0:1] range.

    Updates DataFrame in place.

    Parameters
    ----------
    molecule_df : :pandas:`DataFrame`
        DataFrame of Molecules and properties.

    parameter_columns : :class:`iterable` of :class:`str`
        Column headers for two parameters to plot.
    
    Returns
    -------
    param_headings: `list` of `str`
        parameter headings for the normalised parameters
    
    """
    # Normalise parameters to [0:1] range.
    for param in parameter_columns:
        if len(param.split('-')) == 2:
            std_max = molecule_df[parameter_columns].std().max()
            # molecule_df["norm "+param] = (molecule_df[param]/
                # molecule_df[param].max())
            shift = molecule_df[param].mean() - 2*std_max
            molecule_df["norm "+param] = ((molecule_df[param] - shift)
                                        /(4*std_max))
        elif len(param.split('-')) == 3:
            molecule_df["norm "+param] = (
                molecule_df[param]/180.)
        else:
            molecule_df["norm "+param] = (
                (molecule_df[param]%360.)/360.)

def set_conformer_colours(molecule_df, 
                          quantity_column,
                          colour_map=None):
    """
    Set colours for conformers.

    Colours correspond to quantity if provided otherwise are random.

    Parameters
    ----------
    molecule_df : :pandas:`DataFrame`
        DataFrame of Molecules and properties.

    quantity_column : :class:`str`
        Column header of quantity to colour molecules by.
        [Default: None] If `None` conformers coloured randomly.
    
    Returns
    -------
    colours: :class:`list`
        Colours corresponding to each conformer.
    
    """
    # Calculate normalised energy to plot colour by if given.
    if quantity_column is not None:
        norm_quantity = (molecule_df[quantity_column]
            /molecule_df[quantity_column].max()).values
        
        # Set colour map if not provided.
        if colour_map is None:
            # colour_map = sns.cubehelix_palette(start=2.5, rot=.5, dark=0, light=0.5, as_cmap=True)
            colour_map = sns.cubehelix_palette(light=0.8, as_cmap=True)
        
        # Set colours.
        colours = [colour_map(x)[:3] for x in norm_quantity]    
        return colours
    
    # Else set colours different for each conformer.
    else:
        colours = sns.cubehelix_palette(len(molecule_df))
        return colours

def plot_conf_radar(molecule_df, 
                    parameter_columns, 
                    save=None, 
                    colour=None, 
                    quantity_column=None):
    """
    Plot radial plot of geometric parameters of conformers.

    Parameters
    ----------
    molecule_df : :pandas:`DataFrame`
        DataFrame of Molecules and properties.

    parameter_columns : :class:`iterable` of :class:`str`
        Column headers for two parameters to plot.
    
    save : :class:`str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]

    colour : :class:`list` of :class:`str`
        Colour codes to plot molecules by.
        [Default: `None`] If `None` uses cubehelix colours.

    quantity_column : :class:`str`
        Column header of quantity to colour molecules by.
        [Default: None] If `None` conformers coloured randomly.

    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    # Initialise plot.
    fig, ax = radial_plot_setup()

    # Calculate number of parameters and partition radial plot.
    num_params = len(parameter_columns)
    plot_angles = [n / float(num_params) * 2 * np.pi 
                   for n in range(num_params)]
    plot_angles += plot_angles[:1]

    # Normalise geometric parameters to [0:1] range.
    normalise_parameters(molecule_df, parameter_columns)
    norm_parameter_columns = [('norm '+x) for x in parameter_columns]
    norm_parameter_columns.append(norm_parameter_columns[0])
    
    # Colour conformers by quantity, provided colours, or random.
    if (quantity_column is not None 
        or colour is None):
        molecule_df['colour'] = set_conformer_colours(molecule_df, 
                                                      quantity_column, 
                                                      colour)
    else:
        molecule_df['colour'] = colour

    # Plot for each conformer
    for mol in molecule_df.index:
        ax.plot(plot_angles, molecule_df.loc[mol, norm_parameter_columns],
                label=mol.split('/')[-1], color=molecule_df.loc[mol, 'colour'])
        ax.fill(plot_angles, molecule_df.loc[mol, norm_parameter_columns], 
                color=molecule_df.loc[mol, 'colour'], alpha=0.1)

    # Set plot attributes
    ax.set_xticks(plot_angles[:-1])
    ax.set_xticklabels(parameter_columns)
    ax.set_yticks([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, 
              frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

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


def plot_reaction_profile(reaction_df, 
                          quantity_column=None,
                          save=None, 
                          colour=None, 
                          step_width=3000, 
                          line_buffer=0.08, 
                          label=True,
                          fig=None, 
                          ax=None):
    """
    Plot radial plot of geometric parameters of conformers.

    Parameters
    ----------
    reaction_df : :pandas:`DataFrame`
        DataFrame of Reaction System.

    quantity_column : :class:`str`
        Column header of quantity to plot.
        [Default: None] If `None` plots relative e or g.
    
    save : :class:`str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]

    colour : :class:`list` of :class:`str`
        Colour codes to plot molecules by.
        [Default: `None`] If `None` uses cubehelix colours.

    step_width : :class:`int`
        Width of the horizontal markers for each reaction step.
    
    line_buffer : :class:`float`
        Distance between centre of horizontal marker and
        where connecting line starts.
    
    label : :class:`bool`
        If ``True`` annotates reaction step with molecule name.

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

    # Set number of paths and format settings.
    paths = list(reaction_df['reaction path'].unique())
    label_buffer = line_buffer - 0.01

    # Set colours if not provided. Number of colours is number of paths.
    if colour == None:
        col_pallete = sns.color_palette("cubehelix_r", len(paths))
        col_pallete = sns.cubehelix_palette(len(paths)+1, reverse=True)
        colour = []
        for p_ind in range(len(paths)):
            colour.append(col_pallete[paths.index(p_ind)])

    # Plot horizontal points and connected lines of reaction paths.
    # Line_buffer and step_width can be altered to fit the profile.
    for p_ind, path in enumerate(paths):
        reac_path_df = reaction_df.loc[reaction_df['reaction path'] == path]
        ax.scatter(reac_path_df['rx'], 
                   reac_path_df[quantity_column], 
                   color=colour[p_ind], marker='_', s=step_width, lw=5)
        for rstep_ind in range(1, len(reac_path_df)):
            ax.plot([reac_path_df['rx'].iloc[rstep_ind-1]+line_buffer, 
                    reac_path_df['rx'].iloc[rstep_ind]-line_buffer], 
                    [reac_path_df[quantity_column].iloc[rstep_ind-1], 
                    reac_path_df[quantity_column].iloc[rstep_ind]], 
                    color=colour[p_ind], linestyle='--')

            # Plot labels with dataframe index and energy label.
            if label == True:
                step_label = (str(reac_path_df.index.values[rstep_ind])
                              + ' (' 
                              + str(int(reac_path_df[quantity_column].iloc[rstep_ind]))
                              + ')')
                ax.text(reac_path_df['rx'].iloc[rstep_ind]-label_buffer, 
                        reac_path_df[quantity_column].iloc[rstep_ind]+6, 
                        step_label, color=colour[p_ind], fontsize=11)
        if label == True:
            reactant_label = (str(reac_path_df.index.values[0])
                              + ' (' 
                              + str(int(reac_path_df[quantity_column].iloc[0])) 
                              + ')')
            ax.text(reac_path_df['rx'].iloc[0]-label_buffer, 
                    reac_path_df[quantity_column].iloc[0]+6, 
                    reactant_label, color=colour[p_ind], fontsize=11)

    # Set x and y labels
    ax.set_xlabel('R$_{x}$', fontsize=13)
    ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
    ax.set_xticks([])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

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