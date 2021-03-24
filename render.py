########## INIT ###################################################################################

import numpy as np
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from random import random
import os, sys, pathlib
from math import pi, cos, sin



########## UTILITY FUNCTIONS ######################################################################


def min_max( lst ):
    """ Return a tuple of the minimum and maximum of `lst` """
    return [ np.min( lst ), np.max( lst ) ]


def rand_range( fltMin, fltMax ):
    """ Return a float in between `fltMin` and `fltMax` """
    span = fltMax - fltMin
    return fltMin + random() * span


def total_arrow_len( centers, matx ):
    """ Give the total arrow length between all node centers """
    totLen = 0.0
    N      = len( centers )
    
    for i in range(N):
        for j in range(N):
            if i != j:
                if matx[i,j] != 0.0:
                    vec    = np.subtract( centers[i], centers[j] )
                    totLen += np.linalg.norm( vec )
    return totLen


def get_text_dims( txt, pt=12, aspect=0.5 ):
    """ Imagine how big text would be by rendering it in an invisible figure """
    # NOTE: This is neither an accurate nor a terrible estimate, it seems to work
    fctr = 1/72 # https://en.wikipedia.org/wiki/Point_(typography)
    unit = pt*fctr
    text = str( txt )
    lins = text.splitlines()
    nLin = len( lins )
    wdth = np.max( [len(sub) for sub in lins] )
    wTxt = wdth * unit
    hTxt = nLin * unit * (1/aspect)
    return wTxt, hTxt


########## MarkovDisplay ##########################################################################

class RenderNode():
    
    def __init__(
        self, center, radius, label, 
        facecolor='#2693de', edgecolor='#e6e6e6',
        ring_facecolor='#a3a3a3', ring_edgecolor='#a3a3a3'
        ):
        """
        Initializes a Markov Chain RenderNode(for drawing purposes)
        Inputs:
            - center : RenderNode (x,y) center
            - radius : RenderNode radius
            - label  : RenderNode label
        """
        self.center = center
        self.radius = radius
        self.label  = label

        # For convinience: x, y coordinates of the center
        self.x = center[0]
        self.y = center[1]
        
        # Drawing config
        self.node_facecolor = facecolor
        self.node_edgecolor = edgecolor
        
        self.ring_facecolor = ring_facecolor
        self.ring_edgecolor = ring_edgecolor
        self.ring_width = 0.03  
        
        self.text_args = {
            'ha': 'center', 
            'va': 'center', 
            'fontsize': 16
        }
    
    
    def add_circle(self, ax):
        """
        Add the annotated circle for the node
        """
        circle = mpatches.Circle(self.center, self.radius)
        p = PatchCollection(
            [circle], 
            edgecolor = self.node_edgecolor, 
            facecolor = self.node_facecolor
        )
        ax.add_collection(p)
        ax.annotate(
            self.label, 
            xy = self.center, 
            color = '#ffffff', 
            **self.text_args
        )
        
        
    def add_self_loop(self, ax, prob=None, direction='up', decDig = 5):
        """
        Draws a self loop
        """
        
        arwLen  = 0.30
        theta   = rand_range( -pi*3/8, +pi*3/8 )
        radFctr = self.radius*2.25
        rwHalf  = self.ring_width/2
        
        if direction == 'up':
            start = -30
            angle = 180
            ring_x = self.x
            ring_y = self.y + self.radius
            prob_y = self.y + cos(theta)*radFctr
            x_cent = ring_x - self.radius + rwHalf
            y_cent = ring_y - arwLen
        else:
            start = -210
            angle = 0
            ring_x = self.x
            ring_y = self.y - self.radius
            prob_y = self.y - cos(theta)*radFctr
            x_cent = ring_x + self.radius - rwHalf
            y_cent = ring_y + arwLen
            
        prob_x = self.x + sin(theta)*radFctr
            
        # Add the ring
        ring = mpatches.Wedge(
            (ring_x, ring_y), 
            self.radius, 
            start, 
            angle, 
            width = self.ring_width
        )
        # Add the triangle (arrow)
        offset = 0.1
        left   = [x_cent - offset, ring_y]
        right  = [x_cent + offset, ring_y]
        bottom = [(left[0]+right[0])/2., y_cent]
        arrow  = plt.Polygon([left, right, bottom, left])

        p = PatchCollection(
            [ring, arrow], 
            edgecolor = self.ring_edgecolor, 
            facecolor = self.ring_facecolor
        )
        ax.add_collection(p)
        
        # Probability to add?
        if prob:
            ax.annotate(str( round( prob, decDig ) ), xy=(prob_x, prob_y), color='#000000', **self.text_args)


            
########## MarkovDisplay ##########################################################################


class MarkovDisplay:

    def __init__(self, M, labels, spaceFactor = 6, decDigits = 5):
        """
        Initializes a Markov Chain (for drawing purposes)
        Inputs:
            - M         Transition Matrix
            - labels    State Labels
        """
        
        ## Setup ##
        
        # 0. Set Vars
        self.M        = M # --------- Markov Chain Matrix
        self.n_states = M.shape[0] #- Number of states in the MDP
        self.labels   = labels # ---- State labels
        self.factor   = spaceFactor # Spacing between nodes
        self.decLim   = decDigits # - Number of decimal digits to display

        # 1. Enforce good sizes
        if M.shape[0] < 2:
            raise Exception("There should be at least 2 states")
        if M.shape[0] != M.shape[1]:
            raise Exception("Transition matrix should be square")
        if M.shape[0] != len(labels):
            raise Exception("There should be as many labels as states")
            
        ## Style ##

        # Arrow Style
        self.arrow_facecolor  = '#a3a3a3'
        self.arrow_edgecolor  = '#a3a3a3'
        self.arrow_width      = 0.03
        self.arrow_head_width = 0.20
        
        # RenderNode Style
        self.node_facecolor = '#2693de'
        self.node_edgecolor = '#e6e6e6'
        self.node_radius    = 0.5

        # Text Style
        self.text_args = {
            'ha': 'center',
            'va': 'center',
            'fontsize': 16
        }

        ## Build the network ##
        self.build_network()


    def set_node_centers(self):
        """
        Positions the node centers given the number of states
        """
        # RenderNode positions
        self.node_centers = []
        
        # 0. Find the smallest square number that will contain the nodes
        s = 1
        while s**2 < self.n_states:
            s += 1
        n = s**2
        
        # 1. Assign Centers radiating from upper left
        self.node_centers = [ [ 0, 0 ] ]
        layer = 1
        
        while len( self.node_centers ) < self.n_states:
            lastX = self.node_centers[-1][0]/self.factor
            lastY = self.node_centers[-1][1]/self.factor
            
            if (lastX == 0) and (lastY == 0):
                nuX = 1
                nuY = 0
            
            elif (lastX == layer) and (lastY < layer):
                nuX = lastX
                nuY = lastY+1
            
            elif (lastX == layer) and (lastY == layer):
                nuX = lastX-1
                nuY = lastY
                
            elif (lastX > 0) and (lastY == layer):
                nuX = lastX-1
                nuY = lastY
                
            elif (lastX == 0) and (lastY == layer):
                nuX = layer+1
                nuY = 0
                layer += 1
                
            self.node_centers.append( [nuX*self.factor, nuY*self.factor] )
            
        # 2. Attempt to reduce crossing arrows
        for i in range( self.n_states-1 ):
            for j in range( i+1, self.n_states ):
                # Copy list
                swpLst = list( self.node_centers )
                # Sawp centers
                swpLst[i] = self.node_centers[j]
                swpLst[j] = self.node_centers[i]
                # Compute lengths
                origLen = total_arrow_len( self.node_centers, self.M )
                swapLen = total_arrow_len( swpLst           , self.M )
                # If shorter length, swap order
                if swapLen < origLen:
                    self.node_centers = swpLst
                
        # 3. Set the figure X-Y limits
        self.xlim = ( -self.factor/4.0, np.max( [elem[0] for elem in self.node_centers] ) + self.factor/4.0 )
        self.ylim = ( -self.factor/4.0, np.max( [elem[1] for elem in self.node_centers] ) + self.factor/4.0 )
        
        # 4. Assign figure size
        self.figsize = ( self.xlim[1] + self.factor/4.0 , self.ylim[1] + self.factor/4.0 )
        


    def build_network(self):
        """
        Loops through the matrix, add the nodes
        """
        # Position the node centers
        self.set_node_centers()

        # Set the nodes
        self.nodes = []
        for i in range(self.n_states):
            node = RenderNode(
                self.node_centers[i],
                self.node_radius,
                self.labels[i]
            )
            self.nodes.append(node)


    def add_arrow(self, ax, node1, node2, prob=None):
        """
        Add a directed arrow between two nodes
        """
        # x,y start of the arrow
        x_start = node1.x + np.sign(node2.x-node1.x) * node1.radius
        y_start = node1.y + np.sign(node2.y-node1.y) * node1.radius

        # arrow length
        dx = abs(node1.x - node2.x) - 2.5* node1.radius
        dy = abs(node1.y - node2.y) - 2.5* node1.radius

        # we don't want xoffset and yoffset to both be non-nul
        yoffset = 0.4 * self.node_radius * np.sign(node2.x-node1.x)
        if yoffset == 0:
            xoffset = 0.4 * self.node_radius * np.sign(node2.y-node1.y)
        else:
            xoffset = 0

        arrow = mpatches.FancyArrow(
            x_start + xoffset,
            y_start + yoffset,
            dx * np.sign(node2.x-node1.x),
            dy * np.sign(node2.y-node1.y),
            width = self.arrow_width,
            head_width = self.arrow_head_width
        )
        p = PatchCollection(
            [arrow],
            edgecolor = self.arrow_edgecolor,
            facecolor = self.arrow_facecolor
        )
        ax.add_collection(p)

        DX = dx*np.sign(node2.x-node1.x)
        DY = dy*np.sign(node2.y-node1.y)
        
        probStr = str( round(prob,self.decLim) )
        pW, pH  = get_text_dims( probStr, pt = int(self.text_args['fontsize']), aspect=0.5 )
        
        # Horz
        if dx > dy:
            # Left
            if DX < 0.0:
                yoffset -= pH/2.0
            # Right
            else:
                yoffset += pH/2.0
        
        elif dx == dy:
            
            # Left
            if DX < 0.0:
                yoffset -= pH/2.0
            # Right
            else:
                yoffset += pH/2.0
                
            # Up
            if DY > 0.0:
                xoffset -= pW/2.0
            # Down
            else:
                xoffset += pW/2.0
    
        # Vert
        else:
            # Up
            if DY > 0.0:
                xoffset += pW*0.50
            # Down
            else:
                xoffset -= pW*0.50
                
                
        
        # Probability to add
        frac = rand_range( 0.2, 0.8 )
        x_prob = x_start + xoffset + frac*DX
        y_prob = y_start + yoffset + frac*DY
        if prob:
            ax.annotate( probStr, xy=(x_prob, y_prob), color='#000000', **self.text_args)


    def draw(self, img_path=None):
        """
        Draw the Markov Chain
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Set the axis limits
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

        # Draw the nodes
        for node in self.nodes:
            node.add_circle(ax)

        # Add the transitions
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                # self loops
                if (i == j) and (self.M[i,j] > 0.0):
                    # Loop direction
                    if self.nodes[i].y >= 0:
                        self.nodes[i].add_self_loop(ax, prob = self.M[i,j], direction='up' , decDig = self.decLim)
                    else:
                        self.nodes[i].add_self_loop(ax, prob = self.M[i,j], direction='down' , decDig = self.decLim)
                # directed arrows
                elif self.M[i,j] > 0.0:
                    self.add_arrow(ax, self.nodes[i], self.nodes[j], prob = self.M[i,j])

        plt.axis('off')
        # Save the image to disk?
        if img_path:
            plt.savefig(img_path)
        plt.show()


