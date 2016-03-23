import MySQLdb
import random
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn import svm
import numpy
import pickle
from interface import ClassifierInterface

SCALE = 1000000

def _storeData(data, filename):
	print 'now in storeData'
	with open(filename, 'w') as f:
		pickle.dump(data, f);
def _loadData(filename):
	print 'now in loadData'
	with open(filename, 'r') as f:
		data = pickle.load(f)
	return data

class ICAClassifier(ClassifierInterface):

	def __getGraph(self, listIds, threshold, cur):
		print 'in __getGraph'
		row = []
		col = []
		val = []
		for listId in listIds:
			cur.execute('select user_id from bid where list_id = %d and time <= %f' % (listId, threshold))
			userIds = cur.fetchall()
			userIds = [userId[0] for userId in userIds]
			for userId in userIds:
				row.append(listId)
				col.append(userId)
				val.append(1)

		listToBidGraph = coo_matrix((val, (row, col)))
		listToListGraph = listToBidGraph.dot(listToBidGraph.transpose()).tocoo()
		row = listToListGraph.row
		col = listToListGraph.col
		val = [1.0] * len(row)
		listToListGraph = coo_matrix((val, (row, col)))

		return listToListGraph
	
	def getData(self, **kwargs):
		'''
		input: some metadata for the data you want to get
		output: ((trainVectors, trainLabels, testVectors, testLabels))
		'''
		print 'in getData'
		proportion = kwargs['proportion']
		threshold = kwargs['threshold']

		conn = MySQLdb.connect(kwargs['dbhost'], kwargs['dbuser'], kwargs['dbpw'], kwargs['dbname'])
		cur = conn.cursor()

		queryTemplate = '''
			select
				a.list_id, a.amount, a.max_rate, a.have_house_or_car, a.description_length, sum(b.bid_amount)/a.amount, a.done
			from
				list_attr as a, bid as b
			where
				a.list_id = b.list_id and a.year >= %d and a.year <= %d and a.done = %d and b.time <= %f
			group by
				a.list_id
			order by rand()
			limit %d
			'''
		startYear = 2006
		endYear = 2009
		cur.execute(queryTemplate % (startYear, endYear, 1, threshold, SCALE))
		posLists = cur.fetchall()
		posListsCount = len(posLists)

		cur.execute((queryTemplate) % (startYear, endYear, -1, threshold, posListsCount))
		negLists = cur.fetchall()
		negListsCount = posListsCount

		sampleLists = list(posLists + negLists)
		random.shuffle(sampleLists)

		listIds = [sampleList[0] for sampleList in sampleLists]
		vectors = [sampleList[1:-1] for sampleList in sampleLists]
		labels = [sampleList[-1] for sampleList in sampleLists]
		vectorsCount = 2 * posListsCount

		trainVectorsCount = int(vectorsCount * proportion)

		trainIds = listIds[0:trainVectorsCount]
		trainVectors = vectors[0:trainVectorsCount]
		trainLabels = labels[0:trainVectorsCount]
		testIds = listIds[trainVectorsCount:]
		testVectors = vectors[trainVectorsCount:]
		testLabels = labels[trainVectorsCount:]

		#testListIds = []
		#testVectors = []
		#testLabels = []
		#for index in range(trainVectorsCount, 2 * posListsCount):
		#	if vectors[index][-1] < 1:
		#		testListIds.append(listIds[index])
		#		testVectors.append(vectors[index])
		#		testLabels.append(labels[index])

		scaler = preprocessing.StandardScaler().fit(trainVectors)
		scaledTrainVectors = scaler.transform(trainVectors)
		scaledTestVectors = scaler.transform(testVectors)

		trainGraph = self.__getGraph(trainIds, threshold, cur)
		testGraph = self.__getGraph(testIds, threshold, cur)

		return {'trainVectors': scaledTrainVectors,
				'trainLabels' : trainLabels,
				'trainGraph': trainGraph,
				'trainIds': trainIds,
				'testVectors': scaledTestVectors,
				'testLabels': testLabels,
				'testGraph': testGraph,
				'testIds': testIds,
				}	

	def train(self, data):
		'''
		input:
			data, a dict of data to train(it is from the return value of function 'getData()'
		output:
			a model
		'''	
		print 'in train'
		
		trainVectors = data['trainVectors']
		trainLabels = data['trainLabels']
		trainGraph = data['trainGraph']
		trainIds = data['trainIds']
		
		trainVectors = [list(trainVector) for trainVector in trainVectors]
		trainVectorsCount = len(trainVectors)
		row = [0] * trainVectorsCount
		col = trainIds
		val = [1] * trainVectorsCount
		allOneVector = coo_matrix((val, (row, col)))
		labelVector = coo_matrix((trainLabels, (row, col)))

		neiborCountVector = allOneVector.dot(trainGraph).tocoo()
		neiborWeightVector = labelVector.dot(trainGraph).tocoo() / neiborCountVector
		for index, trainVector in enumerate(trainVectors):
			trainVectors[index].append( float( neiborWeightVector.getcol(trainIds[index]).toarray() ) )

		model = svm.SVC()
		model.fit(trainVectors, trainLabels)

		return model

	def test(self, data, model):
		'''
		input:
			data: a dict of data to test(it is from the return value of function 'getData()'
			model: a model from function 'train()'
		output:
			a tuple of predict labels of the test data.
		'''
		
		print 'in test'
		
		testVectors = data['testVectors']
		testGraph = data['testGraph']
		testIds = data['testIds']
		
		testVectors = [list(testVector) for testVector in testVectors]
		testVectorsCount = len(testVectors)
		row = [0] * testVectorsCount
		col = testIds
		val = [1] * testVectorsCount
		allOneVector = coo_matrix((val, (row, col)))

		neiborCountVector = allOneVector.dot(testGraph).tocoo()

		preOriginalLabels = []
		predictCount = testVectorsCount
		for index, testVector in enumerate(testVectors):
			if testVector[-1] >= 1:
				preOriginalLabels.append(1)
				predictCount -= 1;
			else:
				preOriginalLabels.append(0)
			testVectors[index].append(0)

		originalLabels = preOriginalLabels

		for i in range(20):
			print "in %d iterate" % i
			labelVector = coo_matrix((originalLabels, (row, col)))
			neiborWeightVector = labelVector.dot(testGraph).tocoo() / neiborCountVector
			for index in range(testVectorsCount):
				testVectors[index][-1] = float(neiborWeightVector.getcol(testIds[index]).toarray())
				#print testVectors[index][-1]

			predictLabels = model.predict(testVectors)

			changedCount = 0
			for index in range(testVectorsCount):
				if preOriginalLabels[index] == 1:
					predictLabels[index] = 1;
				else:
					if predictLabels[index] != originalLabels[index]:
						changedCount += 1;

			changedPercentage = 1.0 * changedCount /predictCount
			print "%f changed" % changedPercentage
		
			if changedPercentage <= 0.00001:
				break;

			originalLabels = predictLabels

		return predictLabels

	def evaluate(self, predictLabels, trueLabels):
		print 'in evaluate'
		labelsCount = len(predictLabels)
		correctCount = 0
		for index in range(labelsCount):
			if predictLabels[index] == trueLabels[index]:
				correctCount += 1
		print 'accuracy: %f' % (1.0 * correctCount / labelsCount)


if __name__ == '__main__':
	classifier = ICAClassifier()
	data = classifier.getData(dbhost='localhost', dbname='prosper', dbuser='root', dbpw='3223232mysql', proportion=0.6, threshold=0.6)
	model = classifier.train(data)
	predictLabels = classifier.test(data, model)
	classifier.evaluate(predictLabels, data['testLabels'])

	#storeData(data, 'data.pkl')
	#data = loadData('data.pkl')

	#model = svm.SVC()
	#model.fit(data['trainVectors'], data['trainLabels'])
	#predictLabels = model.predict(data['testVectors'])
	#correctCount = 0;
	#totalCount = len(data['testVectors'])
	#for index in range(totalCount):
	#	if predictLabels[index] == data['testLabels'][index]:
	#		correctCount += 1;
	#print correctCount
	#print totalCount
	#print 1.0 * correctCount / totalCount













