from __future__ import print_function
import httplib2
import os

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import sys
import warnings


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from sklearn import svm
from sklearn import preprocessing

import scalogram
from math import sqrt,pi

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/appsactivity-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/activity https://www.googleapis.com/auth/drive.metadata.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'G Suite Activity API Python Quickstart'

m = []
m2 = []
DELTA = 10
MT = []
cp = []

# cluster 0
points_zero = []			
# cluster 1
points_um = []	
#cluster 2
points_dois = []	
#cluster 3
points_tres = []
# cluster 4
points_quatro = []	
# cluster 5	
points_cinco = []
dist_c1= []

# cluster 0
points_zeroS = []			
# cluster 1
points_umS = []	
#cluster 2
points_doisS = []	
#cluster 3
points_tresS = []
# cluster 4
points_quatroS = []	
# cluster 5	
points_cincoS = []


def waitforEnter():
	if sys.version_info[0] == 2:
		raw_input("Press ENTER to continue.")
	else:
		input("Press ENTER to continue.")

def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'appsactivity-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials



def get_activity():
	credentials = get_credentials()
	http = credentials.authorize(httplib2.Http())
	service = discovery.build('appsactivity', 'v1', http=http)

	list_time = []
	list_time2 = []
	list_action = []
	list_action2 = []

	#page_token = None 
	#while True:
	results = service.activities().list(source='drive.google.com', groupingStrategy = 'none',
					drive_ancestorId='root', pageSize=35).execute()

	activities = results.get('activities', [])
	
	if not activities:
		print('No activity.')
	else:
		print('Recent activity:')
		
	cenas = int(activities[0]['combinedEvent']['eventTimeMillis'])/1000
	for activity in activities:
		event = activity['combinedEvent']
		user = event.get('user', None)
		target = event.get('target', None)
		
		if user == None or target == None:
			continue
			
				
		timestamp = int(event['eventTimeMillis'])/1000
		time = datetime.datetime.fromtimestamp(timestamp)
		
		if cenas - DELTA < timestamp:
			list_time.append(timestamp)
			list_action.append(event['primaryEventType'])
			
		else:
			#print(list_time)
			#print(list_action)
			#print("---------------------")
			processing(list_action,timestamp)
			list_time = []
			list_action = []
			cenas = timestamp
			
			
		print('{0}: {1}, {2}, {3} ({4})'.format(time, user['name'].encode('utf-8'),
			event['primaryEventType'].encode('utf-8'), target['name'].encode('utf-8'), target['mimeType'].encode('utf-8')))
			
	MT = np.array(m)
	print(len(MT))

	#np.savetxt('Profiling2.out',MT,fmt='%-7.2f')
	
	return MT
	
		
def processing(list_action,timestamp):
	e = 0
	t = 0
	r = 0
	u = 0
	move = 0
	
	time = datetime.datetime.fromtimestamp(timestamp)
	
	for i in list_action:
		if i == 'edit':
			e = e +1	
		elif i =='trash':
			t = t + 1
		elif i == 'rename':
			r = r +1
		elif i == 'move':
			move = move +1
		elif i == 'upload':
			u = u +1
		
	m.append([e,u,r,move,t,time.hour])

##############################################################################################################
# Intervalo de tempo 10s
#Contagens
###############################################################################################################
def cluster_Profiling():
	data = np.loadtxt('Profiling2.out')

	
	
	#data = preprocessing.normalize(data)
	
	#print(data)
	
	
	edits = data[:,0]
	up = data[:,1]
	r = data[:,2]
	move = data[:,3]
	trash = data[:,4]
	time = data[:,5]
	
	features=np.c_[edits,up,r,move,trash,time] 
	rcp = PCA(n_components=2).fit_transform(features)

	#K-means assuming 2 clusters
	kmeans = KMeans(init='k-means++', n_clusters=6)
	kmeans.fit(rcp)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	
	'''
		Calcular o raio: > distncia de um ponto ao centro
		Calcular perimetro
	
	'''
	zero = []
	um = []
	dois = []
	tres = []
	quatro = []
	cinco = []
	
	for i in range(len(labels)):
		if labels[i] == 0:
			zero.append(i)
		elif labels[i] ==1:
			um.append(i)
		elif labels[i] == 2:
			dois.append(i)
		elif labels[i] == 3:
			tres.append(i)
		elif labels[i] == 4:
			quatro.append(i)
		else:
			cinco.append(i)
			
		
	c_i = [zero, um, dois, tres, quatro, cinco]

	c_o = [points_zero,points_um, points_dois, points_tres, points_quatro, points_cinco]		
		
	for i in range(len(c_i)):	
		Cluster_points(rcp, c_i[i],c_o[i])

	
	dist0 = []
	for i in range(0, len(points_zero)):
		rcp_x, rcp_y= points_zero[i]
		c_x, c_y = centroids[0]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist0.append(d)
		
	dist1 = []
	for i in range(0, len(points_um)):
		rcp_x, rcp_y = points_um[i]
		c_x, c_y = centroids[1]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist1.append(d)
		
	dist2= []
	for i in range(0, len(points_dois)):
		rcp_x, rcp_y = points_dois[i]
		c_x, c_y = centroids[2]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist2.append(d)
		
	dist3 = []
	for i in range(0, len(points_tres)):
		rcp_x, rcp_y = points_tres[i]
		c_x, c_y = centroids[3]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist3.append(d)
		
	dist4 = []
	for i in range(0, len(points_quatro)):
		rcp_x, rcp_y = points_quatro[i]
		c_x, c_y = centroids[4]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist4.append(d)
		
	dist5 = []
	for i in range(0, len(points_cinco)):
		rcp_x, rcp_y = points_cinco[i]
		c_x, c_y = centroids[5]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
		dist5.append(d)
	
	print(len(dist0),len(points_zero), len(zero))
	
	#get max dist -> equals ratio!!!!
	d = [dist0, dist1, dist2, dist3, dist4, dist5]
	max_dist = []
	for i in range(len(d)):
		max_dist.append(max(d[i]))
		
	# give margin to system progress
	# max(dist) * 15 %
	new_maxdist = [i * 1.20 for i in max_dist]
	matrix_clusterDist = np.column_stack((np.arange(6),new_maxdist))
	print(matrix_clusterDist)
	
	'''
		Save radius
	'''
	np.savetxt('Radius.out',matrix_clusterDist,fmt='%-7.2f')
	
	# get perimeter
	#perimeter = 2 * pi * max_dist
	#print(perimeter) 

		
	np.savetxt('Centroids.out',centroids,fmt='%-7.2f')
	
	
	

def new_data():
	#matrix = get_activity()
	#print(matrix)
	
	
	#matrix = np.loadtxt('boot_rename.txt')
	
	#matrix = np.loadtxt('boot_move.txt')
	#matrix = np.loadtxt('boot_rename.txt')
	#matrix = np.loadtxt('boot_delete.txt')
	matrix = np.loadtxt('normal.out')
	#matrix = np.loadtxt('boot.out')
	#matrix = preprocessing.normalize(matrix)
	
	edits = matrix[:,0]
	up = matrix[:,1]
	r = matrix[:,2]
	move = matrix[:,3]
	trash = matrix[:,4]
	time = matrix[:,5]
	
	features=np.c_[edits,up,r,move,trash,time] 	 
	rcp = PCA(n_components=2).fit_transform(features)

	#K-means assuming 2 clusters
	kmeans = KMeans(init='k-means++', n_clusters=6)
	kmeans.fit(rcp)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	
	
	
	#Z = kmeans.predict(rcp)
	#print('Predict', Z)
	
	read_centroids = np.loadtxt('Centroids.out')
	read_radius = np.loadtxt('Radius.out')
	
	dist = []
	
	for i in range(0, len(rcp)):
		rcp_x, rcp_y = rcp[i]
		for j in range(0,len(read_centroids)):
			c_x, c_y = read_centroids[j]
			d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2)
			dist.append(d)

	t = np.array(dist)
	M_dist = np.reshape(t,(len(rcp),len(read_centroids)))
	
	#print('\nM_dist',M_dist)
	
	c = []
	v = []
	
	# Calculate min distance for each centroid
	for i in range(0,len(M_dist)):
		centro = np.argmin(M_dist[i])
		value_min = min(M_dist[i])
		c.append(centro)
		v.append(value_min)
		
	b = np.column_stack((c,v))
	
	print(b)
	
	
	cluster = b[:,0]
	dist_cluster = b[:,1]
	
	cluster_radius = read_radius[:,0]
	radius = read_radius[:,1]
	
	anomalie = []
	normal = []
	
	for i in range(len(b)):
		if cluster[i] in  cluster_radius:
			if dist_cluster[i] > radius[cluster[i]]:
				anomalie.append(i)
			else:
				normal.append(i)
							
	'''
		Quando nao se verifica anomalia, pontos sao guardados no ficheiro 
		da ativiade normal
	'''
	
	print('anomalie:',anomalie)
	
	# make list with labels
	
	#l = np.arange(0,len(matrix),1)
	#print(list(l))
	#ll =list(l)
	
	true = 0
	for i in range(len(anomalie)):
		if anomalie[i] in range(len(matrix)):
			true = true + 1
	print(true)
	t = len(matrix)-1
	acc = float(true)/t * 100
	print(acc)
	
	malicios_points = []
	normal_points = []
	
	if len(anomalie) == 0:
		# all points are Normal
		with open("Profiling2.out", "a") as myfile:
			np.savetxt(myfile,matrix,fmt='%-7.2f')
	else:
		# may be exits anomalie and normal points
		for i in range(len(anomalie)):
			malicios_points.append(matrix[anomalie[i]])
			
		for i in range(len(normal)):
			normal_points.append(matrix[normal[i]])
			
		if len(normal_points)> 0:
			with open("Profiling2.out", "a") as myfile:
				np.savetxt(myfile,normal_points,fmt='%-7.2f')
				
		if len(malicios_points) > 0:		
			if not os.path.isfile('Anomalies.out'):
				np.savetxt('Anomalies.out',malicios_points,fmt='%-7.2f')
			else:
				with open("Anomalies.out", "a") as myfile:
					np.savetxt(myfile,malicios_points,fmt='%-7.2f')
	
	
############################################################################################################################


##############################################################################################################
# Intervalo de tempo 100s
#Estatisticas
###############################################################################################################

'''	
  Delta = 100 s	
  Agrupa 10 linhas da matrix com delta = 10 s
'''
def process_stats(m,int1, int2, nr_f):
	M = []
	MM = []
	V = []
	S = []	
	DT = int1/int2
	
	MX = []	
		
	i =DT
	j = 0
	
	data = np.loadtxt('Profiling2.out')
	
	if m != None:
		matrix = m
	else:
		matrix = data
	
	while i <= len(matrix):	
		M.append(np.mean(matrix[j:i,:],axis = 0))
		MM.append(np.median(matrix[j:i,:],axis = 0))
		V.append(np.var(matrix[j:i,:],axis = 0))
		S.append(stats.skew(matrix[j:i,:]))		
		i+=DT
		j+=DT
		 
	i=DT
	j=0
		
	MX = np.concatenate((M,MM,V,S), axis = 1)

	
	print('\nMX ', MX)
	
	return MX

	
def Profiling_Stats(matrix):
	
	#features for training
	#Medias = matrix[0]
	
	#print('\nMedias',Medias)
	
	#medianas = matrix[1]
	#var = matrix[2]
	#S = matrix[3]
	
	# ATENCAO: PCA de 3 componentes
	features=np.c_[matrix] 
	rcp = PCA(n_components=3).fit_transform(features)
	
	#K-means assuming 3 clusters
	kmeans = KMeans(init='k-means++', n_clusters=6)
	kmeans.fit(rcp)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	
	'''
		Calcular o raio: > distncia de um ponto ao centro
		Calcular perimetro
	
	'''
	
	
	zero = []
	um = []
	dois = []
	tres = []
	quatro = []
	cinco = []
	
	for i in range(len(labels)):
		if labels[i] == 0:
			zero.append(i)
		elif labels[i] ==1:
			um.append(i)
		elif labels[i] == 2:
			dois.append(i)
		elif labels[i] == 3:
			tres.append(i)
		elif labels[i] == 4:
			quatro.append(i)
		else:
			cinco.append(i)
			
	
	c_i = [zero, um, dois, tres, quatro, cinco]


	c_o = [points_zero,points_um, points_dois, points_tres, points_quatro, points_cinco]		
		
	for i in range(len(c_i)):	
		Cluster_points(rcp, c_i[i],c_o[i])
	
	dist0 = []
	for i in range(0, len(points_zero)):
		rcp_x, rcp_y, rcp_z = points_zero[i]
		c_x, c_y, c_z = centroids[0]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist0.append(d)
	
	dist1 = []
	for i in range(0, len(points_um)):
		rcp_x, rcp_y,rcp_z = points_um[i]
		c_x, c_y, c_z = centroids[1]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist1.append(d)
		
	dist2= []
	for i in range(0, len(points_dois)):
		rcp_x, rcp_y, rcp_z = points_dois[i]
		c_x, c_y, c_z = centroids[2]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist2.append(d)
		
	dist3 = []
	for i in range(0, len(points_tres)):
		rcp_x, rcp_y, rcp_z = points_tres[i]
		c_x, c_y, c_z = centroids[3]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist3.append(d)
		
	dist4 = []
	for i in range(0, len(points_quatro)):
		rcp_x, rcp_y, rcp_z = points_quatro[i]
		c_x, c_y, c_z = centroids[4]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist4.append(d)
		
	dist5 = []
	for i in range(0, len(points_cinco)):
		rcp_x, rcp_y, rcp_z = points_cinco[i]
		c_x, c_y, c_z = centroids[5]
		d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
		dist5.append(d)
	
	
	#get max dist -> equals ratio!!!!
	d = [dist0, dist1, dist2, dist3, dist4, dist5]
	max_dist = []
	for i in range(len(d)):
		max_dist.append(max(d[i]))
		
	# give margin to system progress
	# max(dist) * 15 %
	new_maxdist = [i * 1.20 for i in max_dist]
	matrix_clusterDist = np.column_stack((np.arange(6),new_maxdist))
	
	print('MAX',matrix_clusterDist)
	
	'''
		Save radius
	'''
	np.savetxt('Radius_Stats.out',matrix_clusterDist,fmt='%-7.2f')
	
	# get perimeter
	#perimeter = 2 * pi * max_dist
	#print(perimeter) 

		
	np.savetxt('Centroids_Stats.out',centroids,fmt='%-7.2f')
	
	
def new_data_stats():
	
	matrix = np.loadtxt('activity.txt')
	#matrix = np.loadtxt('boot_move.txt')
	#matrix = np.loadtxt('boot_rename.txt')
	#matrix = np.loadtxt('boot_delete.txt')
	
	
	MT = process_stats(matrix,100,10,6)
	
	#Medias = MT[0]
	#medianas = MT[1]
	#var = MT[2]
	#S = MT[3]

	# ATENCAO: PCA de 3 componentes
	features=np.c_[MT] 
	rcp = PCA(n_components=3).fit_transform(features)
	
	#K-means assuming 3 clusters
	kmeans = KMeans(init='k-means++', n_clusters=6)
	kmeans.fit(rcp)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	
	
	
	#Z = kmeans.predict(rcp)
	#print('Predict', Z)
	
	read_centroids = np.loadtxt('Centroids_Stats.out')
	read_radius = np.loadtxt('Radius_Stats.out')
	
	dist = []
	
	for i in range(0, len(rcp)):
		rcp_x, rcp_y , rcp_z = rcp[i]
		for j in range(0,len(read_centroids)):
			c_x, c_y, c_z= read_centroids[j]
			d = sqrt((rcp_x - c_x)**2 + (rcp_y - c_y)**2 + (rcp_z - c_z)**2)
			dist.append(d)

	t = np.array(dist)
	M_dist = np.reshape(t,(len(rcp),len(read_centroids)))
	
	print('\nM_dist',M_dist)
	
	c = []
	v = []
	
	# Calculate min distance for each centroid
	for i in range(0,len(M_dist)):
		centro = np.argmin(M_dist[i])
		value_min = min(M_dist[i])
		c.append(centro)
		v.append(value_min)
		
	b = np.column_stack((c,v))
	
	print('TK',b)
	
	
	cluster = b[:,0]
	dist_cluster = b[:,1]
	
	cluster_radius = read_radius[:,0]
	radius = read_radius[:,1]
	
	anomalie = []
	normal = []
	
	for i in range(len(b)):
		if cluster[i] in  cluster_radius:
			if dist_cluster[i] > radius[cluster[i]]:
				anomalie.append(i)
			else:
				normal.append(i)
							
	'''
		Quando nao se verifica anomalia, pontos sao guardados no ficheiro 
		da ativiade normal
	'''
	
	print('anomalie:',anomalie)
	malicios_points = []
	normal_points = []
	
	
	if len(anomalie) == 0:
		# all points are Normal
		with open("Profiling2.out", "a") as myfile:
			np.savetxt(myfile,matrix,fmt='%-7.2f')
	else:
		# may be exits anomalie and normal points
		for i in range(len(anomalie)):
			malicios_points.append(matrix[anomalie[i]])
			
		for i in range(len(normal)):
			normal_points.append(matrix[normal[i]])
			
		if len(normal_points)> 0:
			with open("Profiling2.out", "a") as myfile:
				np.savetxt(myfile,normal_points,fmt='%-7.2f')
				
		if len(malicios_points) > 0:		
			if not os.path.isfile('Anomalies.out'):
				np.savetxt('Anomalies.out',malicios_points,fmt='%-7.2f')
			else:
				with open("Anomalies.out", "a") as myfile:
					np.savetxt(myfile,malicios_points,fmt='%-7.2f')
	
	
	
	
##############################################################################################################
# Intervalo de tempo 600s
# Scalogram
###############################################################################################################

def scalograma(matrix):
	# Aumentar a escala de tempo	
	
	picos = []
	scales=np.arange(1,50)
	allS=np.zeros((10,len(scales)))
	
	DT = 10
	
	i =DT
	j = 0
	
	for k in range(6):
		while i <= len(matrix):	
			S,scales=scalogram.scalogramCWT(matrix[j:i,k],scales)
			
			#allS[i,:]=S
			#plt.plot(scales,S,'b')
			#plt.show()
			idx=argrelextrema(S, np.greater)[0]
			threshold=.5*max(S)
			idxM=[p for p in idx if S[p]>threshold]
			print('idxM',idxM)
			print('scales',scales[idxM])
			# Retornar os picos (xx) maiores que o treshold
			picos.append(scales[idxM])
			
			waitforEnter()
			
			i+=DT
			j+=DT
			
		i=DT
		j=0
	
	# [] importam?????	
	#picos = [x for x in picos if x != []]		

	return picos


###################################################################################


def Cluster_points(rcp, list_in, list_out):
	for j in range(len(list_in)):
		list_out.append(rcp[list_in[j]])



		
def main():
	
	''' DELTA = 30 s without statics'''
	cluster_Profiling()
	new_data()
	
	''' DELTA = 100 s with statics '''
	#matrix_stats = process_stats(None,100, 10, 6)
	#Profiling_Stats(matrix_stats)
	#new_data_stats()
	
	#get_activity()
	
	''' DELTA = 600 s for scalogram '''
	#picos = scalograma(matrix)
	#matrix_class = classif(matrix)
	#statics(matrix_class)	
	
		
if __name__ == '__main__':
    main()
