import unittest

from data_io.sick import Sick_Data


class TestSickDataLoader(unittest.TestCase):

    def setUp(self):
        self.sickdataloader_train = Sick_Data('/home/meyerjo/dataset/contrastive/sick/20210607_20210624/')
        self.sickdataloader_test = Sick_Data('/home/meyerjo/dataset/contrastive/sick/20210607_20210624/', train=False)

    def testLength(self):
        self.assertEqual(len(self.sickdataloader_train), 1733)
        self.assertEqual(len(self.sickdataloader_test), 191)


    def testAccess(self):
        data = self.sickdataloader_train[0]
        print(data)