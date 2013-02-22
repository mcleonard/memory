# Unit Catalog

# Decided to use SQLAlchemy for this

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.exc import *
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy import ForeignKey, create_engine, and_
from sqlalchemy.orm import relationship, backref, sessionmaker

#################################################
# Make all your classes here

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    project = Column(String)
    researcher = Column(String)
    
    sessions = relationship("Session", order_by="Session.id", backref="project")
    
    def __init__(self, project, researcher = None):
        self.project = project
        self.researcher = researcher
    
    def __repr__(self):
        return "<Project('%s')>" % self.project

class Session(Base):
    '''
    project : which project/task the unit was recorded for
    rat : the name of the rat the unit came from
    date : date unit was recorded
    '''
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    
    rat = Column(String)
    date = Column(String)
    notes = Column(String)
    duration = Column(Float)
    depth = Column(Float)
    path = Column(String)
    
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    units = relationship("Unit", order_by="Unit.id", backref="session")
    
    def __init__(self, project, rat, date):
        self.project = project
        self.rat = rat
        self.date = date
    
    def __repr__(self):
        return "<Session(%s, %s)>" % (self.rat, self.date)

class Unit(Base):
    '''
    
    tetrode : tetrode number unit was recorded on
    cluster : cluster number of the unit from the sorting
    datapath : path to the folder containing all the data
    falsePositive : false positive rate, from cluster metrics
    falseNegative : false negative rate, from cluster metrics
    notes : any notes about the unit
    depth : depth at which the unit was recorded
    
    Just a warning, documenation of the attributes might be wrong,
    but the code should always work.
    
    '''
    __tablename__ = 'units'
    
    id = Column(Integer, primary_key=True)

    tetrode = Column(Integer)
    cluster = Column(Integer)
    falsePositive = Column(Float)
    falseNegative = Column(Float)
    notes = Column(String)
    rate = Column(Float)
    
    session_id = Column(Integer, ForeignKey("sessions.id"))
    
    def __init__(self, session, tetrode, cluster):
        self.session = session
        self.tetrode = tetrode
        self.cluster = cluster
    
    def __repr__(self):
        return "<Unit(%s, %s,tetrode %s,cluster %s)>" \
            % (self.id, self.session, self.tetrode, self.cluster)

####################################################################################

class ExistsError(Exception):
    pass
    
####################################################################################

# Actual catalog object here

class Catalog(object):
    
    def __init__(self, database, echo = False):
        
        self.engine = create_engine('sqlite://'+database, echo = echo)
        self._ConnectGen = sessionmaker(bind=self.engine)
        self.connection = self._ConnectGen()
        
        Base.metadata.create_all(self.engine)
    
    def open(self):
        ''' Returns a connection to the database, an SQLAlchemy session generated
            by a sessionmaker bound to self.engine '''
        
        connection = self._ConnectGen()
        
        return connection
    
    def start_project(self, project, researcher = None):
        ''' Start a project and add it to the database '''
        
        conn = self.open()
        conn.add(Project(project, researcher))
        conn.commit()
    
    def get_project(self, project):
        ''' Returns a Project
        
        Parameters
        ----------
        project : string
            name of the project you want
        
        Returns
        -------
        out : Project object
            the requested Project
        
        '''      
        
        conn = self.connection
        
        try:
            proj = conn.query(Project).filter(Project.project == project).one()
        except NoResultFound:
            raise NoResultFound("Project %s doesn't exist, create it first." % project)
            
        return proj
    
    def get_session(self, rat, date, connection=None):
        ''' Returns a Session from the specified rat and date'''
        
        conn = self.connection
    
        try:
            session = conn.query(Session).filter(Session.rat==rat).\
                filter(Session.date==date).one()
        except NoResultFound:
            raise NoResultFound("Session doesn't exist")
        
        return session
    
    def add_unit(self, project, rat, date, tetrode, cluster, **kwargs):
        ''' Add a unit to a project '''
        
        # First, we'll check if the unit already exists
        try:
            unit = self.get_unit(rat, date, tetrode, cluster)
            raise ExistsError("Unit already exists")
        except NoResultFound:
            pass
        
        conn = self.connection
        
        proj = self.get_project(project)
        
        try:
            session = self.get_session(rat, date)
        except NoResultFound:
            session = Session(proj, rat, date)
        
        unit = Unit(session, tetrode, cluster)
        conn.commit()
        
        if kwargs:
            self.update_unit(unit, **kwargs)
            
    def add_batch(self, unit_list):
        self.connection.add_all(unit_list)
        self.connection.commit()
    
    def get_unit(self, *args, **kwargs):
        '''
        Returns unit given by the unit id, or by the rat name, date, tetrode, cluster.
        
        Parameters
        ----------
        
        unit_id : int
            the unit id of the unit you want
        
        or
        
        rat : str
            the name of the rat
        date : str
            the date the unit was recorded
        tetrode : int
            the tetrode the unit is from
        cluster : int
            the cluster from the tetrode
        '''
        
        conn = self.connection
    
        valid_types = [type(''), type(''), type(1), type(1)]
        arg_types = [ type(arg) for arg in args ]
        
        if (len(args)==1) & (type(args[0]) == type(1)):
            # Assuming the argument is a unit id
            unit = conn.query(Unit).filter(Unit.id == args[0]).one()
        elif (len(args)==4) & (arg_types == valid_types):
            # Assuming the argument is of the form [rat, date, tetrode, cluster]
            rat = args[0]
            date = args[1]
            tetrode = args[2]
            cluster = args[3]
            
            unit = conn.query(Unit).join(Session).\
                filter(Session.rat == rat).filter(Session.date == date).\
                filter(Unit.tetrode == tetrode).filter(Unit.cluster == cluster).\
                one()
        elif (len(args)==4) & (arg_types != valid_types):
            raise ValueError('Input arguments do not conform to valid types')
        else:            
            raise ValueError('len(args) = %s, must be 1 or 4' % len(args))
            
        return unit
        
    def __getitem__(self, key):
        if isinstance(key, type(1)):
            return self.get_unit(key)
        elif (type(key) == type([])):
            return [self[kk] for kk in key]
        else:
            raise KeyError("Key must be an integer or list of integers")

    def update_unit(self, *args, **kwargs):
        ''' Update a unit already in the database 
        
        Parameters
        ----------
        args:
        
        unit_id : int
            the unit id of the unit you want
        
        or
        
        rat : str
            the name of the rat
        date : str
            the date the unit was recorded
        tetrode : int
            the tetrode the unit is from
        cluster : int
            the cluster from the tetrode
            
        or 
        
        unit : Catalog.Unit object
            the Unit object you want to update
        
        >>KEYWORDS<<
        The attributes of Unit
        path : path to the folder containing all the data
        falsePositive : false positive rate, from cluster metrics
        falseNegative : false negative rate, from cluster metrics
        notes : any notes about the unit
        depth : depth at which the unit was recorded
        
        The keywords in the documentation might not be updated along
        with the attributes of the Unit class.  However, the valid keywords
        in the code will always update with the Unit class attributes, so any
        attribute of the Unit class will be available through this method.
        
        '''
        
        # Doing it this way so that if the attributes of the Unit or Session
        # classes are changed, the valid keys here will change automatically
        valid_unit_keys = [key for key in dir(Unit) if key[0]!='_']
        excise = ['metadata', 'id', 'session', 'session_id']  # don't want these to be accessed
        for exc in excise:
            valid_unit_keys.remove(exc)
        
        valid_session_keys = [key for key in dir(Session) if key[0]!='_']
        excise = ['metadata', 'id', 'units', 'project_id']  # don't want these to be accessed
        for exc in excise:
            valid_session_keys.remove(exc)
        
        if type(args[0]) == Unit:
            unit = args[0]
        else:
            unit = self.get_unit(*args)
        
        session = unit.session
        
        if kwargs:
            for key in kwargs.iterkeys():
                if key in valid_unit_keys:
                    setattr(unit, key, kwargs[key])
                elif key in valid_session_keys:
                    setattr(session, key, kwargs[key])
                else:
                    raise NameError('%s not a valid keyword' % key)
        
    def query(self,*args, **kwargs):
        ''' Just to make things a little easier'''
        return self.connection.query(*args,**kwargs)

def calculate_rates(all_units):
    
    import os
    import cPickle as pkl
    
    sessions = np.unique([ unit.session for unit in all_units ])

    for session in sessions:

        units = session.units
        tetrodes = np.unique([ unit.tetrode for unit in units ])
        path = os.path.expanduser(session.path)
        
        for tetrode in tetrodes:
            # Load the data
            filename = '%s_%s.cls.%s' % (session.rat, session.date, tetrode)
            path_filename = os.path.join(path, filename)
            with open(path_filename,'r') as cls:
                clusters = pkl.load(cls)
            
            units = [ unit for unit in session.units if unit.tetrode == tetrode ]
            
            for unit in units:
                unit_data = clusters[unit.cluster]
                if type(unit_data['peaks']) == type(np.array(1)):
                    spike_count = len(unit_data['peaks'])
                else:
                    spike_count = 0
                unit.rate = spike_count / session.duration