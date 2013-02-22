# Unit tests for catalog.py

import unittest
from catalog import *
from sqlalchemy.orm.exc import *

class CatalogTester(unittest.TestCase):
    
    def setUp(self):
        self.catalog = Catalog('/:memory:')
        self.project = 'LLRR'
    
    def testAddProject(self):
        
        researcher = 'Bob'
        self.catalog.start_project(self.project, researcher)
        
    def testGetProject(self):
        
        self.catalog.start_project(self.project)
        
        proj = self.catalog.get_project(self.project)
        self.assertTrue(proj.project == self.project)
        
        with self.assertRaises(NoResultFound):
             self.catalog.get_project('')
        
    def testAddUnit(self):
        
        self.catalog.start_project(self.project)
        
        self.catalog.add_unit(self.project, 'Rat', 'Date', 1, 1)
        
        with self.assertRaises(ExistsError):
            self.catalog.add_unit(self.project, 'Rat', 'Date', 1, 1)
            
        self.catalog.add_unit(self.project, 'Rat2', 'Date', 2, 3, falsePositive=0.04, path = '/path/to/data')
        
    def testGetUnit(self):
        
        self.catalog.start_project(self.project)
        self.catalog.add_unit(self.project, 'Rat', 'Date', 1, 1)
        
        unit = self.catalog.get_unit('Rat', 'Date', 1, 1)
        
        self.assertEqual(unit.session.rat, 'Rat')
        self.assertEqual(unit.session.date, 'Date')
        self.assertEqual(unit.tetrode, 1)
        self.assertEqual(unit.cluster, 1)
        
        self.catalog.get_unit(unit.id)
        
        with self.assertRaises(NoResultFound):
            self.catalog.get_unit('', '', 2, 3)
        
    
    def testUpdateUnit(self):
        
        kwarg = {'falsePositive':0.05, 'falseNegative':0.04, 'notes': 'A lovely neuron', 'depth':2.1}
        self.catalog.start_project(self.project)
        self.catalog.add_unit(self.project, 'Rat', 'Date', 1, 1)
        unit = self.catalog.get_unit('Rat', 'Date', 1, 1)
        
        self.catalog.update_unit(unit, **kwarg)
        self.assertEqual(unit.falsePositive, kwarg['falsePositive'])
        self.assertEqual(unit.falseNegative, kwarg['falseNegative'])
        self.assertEqual(unit.notes, kwarg['notes'])
        self.assertEqual(unit.session.depth, kwarg['depth'])
        
        self.catalog.update_unit(unit.id, **kwarg)
        self.assertEqual(unit.falsePositive, kwarg['falsePositive'])
        self.assertEqual(unit.falseNegative, kwarg['falseNegative'])
        self.assertEqual(unit.notes, kwarg['notes'])
        self.assertEqual(unit.session.depth, kwarg['depth'])
    
    def testBatchAdd(self):
        
        unit_list = batchUnits(self.project, dir)
        
        self.start_project(self.project)
        self.catalog.add_batch(unit_list)
        
        for unit in unit_list:
            got = self.catalog.get_unit(unit.id)
            self.assertEqual(got, unit)
            
        
if __name__ == '__main__':
    unittest.main()