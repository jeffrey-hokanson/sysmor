#provides a class for dumping tab separated files for PGF

class PGF:
	def __init__(self):
		self.column_names = []
		self.columns = []

	def add(self, name, column):
		if len(self.columns) > 1:
			assert len(self.columns[0]) == len(column)

		self.columns.append(column)
		self.column_names.append(name)

	def keys(self):
		return self.column_names

	def __getitem__(self, key):
		i = self.column_names.index(key)
		return self.columns[i]

	def write(self, filename):
		f = open(filename,'w')

		for name in self.column_names:
			f.write(name + '\t')
		f.write("\n")		

		for j in range(len(self.columns[0])):
			for col in self.columns:
				f.write("{}\t".format(float(col[j])))
			f.write("\n")

		f.close()

	def read(self, filename):
		with open(filename,'r') as f:
			for i, line in enumerate(f):
				# Remove the newline and trailing tab if present
				line = line.replace('\t\n','').replace('\n','')
				if i == 0:
					self.column_names = line.split('\t')
					self.columns = [ [] for name in self.column_names]
				else:
					cols = line.split('\t')
					for j, col in enumerate(cols):
						self.columns[j].append(float(col))


