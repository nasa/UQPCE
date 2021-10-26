#*******************************************************************************
#                                                                              *
# Created by:                 Joanna Schmidt                                   *
#                               2019/11/27                                     *
#                                                                              *
# Helper functions for the UQPCE wrapper.                                      *
#                                                                              *
#*******************************************************************************

from collections import namedtuple

Inputs = namedtuple(
    'Inputs', ['altitude', 'ground_course', 'gspeed', 'roll', 'pitch', 'yaw',
               'wind_vel', 'wind_dir']
)


def in_dict(key, dictionary):
    """
    Inputs:     key- the key to be checked for in the dictionary
                dictionary- the dictionary the key will be searched in
                
    Checks a dictionary to see if it contains a key. If it does, True is 
    returned; False is returned otherwise.
    """
    keys = list(dictionary)

    for each in keys:
        if each == key:
            return True

    return False
