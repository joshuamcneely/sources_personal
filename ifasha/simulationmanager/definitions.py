#!/usr/bin/env python

class Type:
    """
    3 cases long no no lower score"""
    def __init__(self):
        self.dynamic = 'dyn'
        self.static = 'sts'
        self.mine = 'nd0'

class Status:
    def __init__(self):
        self.status = 'status'
        self.initial = 'init'
        self.generated = 'gen'
        self.submitted = 'PD' # pending
        self.running = 'R'
        self.finished = 'FN'
        self.postprocessed = 'DONE'
        self.broken = 'BROKEN'
        self.order = {
            self.initial : 0,
            self.generated : 1,
            self.submitted :2,
            self.running : 3,
            self.finished : 4,
            self.postprocessed : 5,
            self.broken : 6,
        }

class Time:
    def __init__(self):
        self.when = {
            'init'  : 'initial_time',
            'R'  : 'start_time',
            'FN' : 'finish_time',
        }
        self.initial = 'initial_time'
        self.start = 'start_time'
        self.finish = 'finish_time'


class Table:
    def __init__(self):
        self.simulation  = 'simulation'
        self.input    = 'input'
        self.options     = 'options'
        self.analysis = 'analysis'
        self.all = [
            self.simulation,  
            self.analysis,
            self.input, 
            self.options,
        ]

     
            
            
