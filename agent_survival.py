import csv
import pandas as pd
import numpy as np
import string
import sklearn
from sklearn.neural_network import MLPClassifier
from patsy import dmatrices
from sklearn import metrics
import random
import logging
from tqdm import tqdm
from functools import reduce
from operator import add
from collections import OrderedDict
from math import floor
import math
from pybrain.structure import FeedForwardNetwork, SigmoidLayer, FullConnection, LinearLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
import pygame
import pybrain
import time
from statistics import mean

''' initialize pygame and screen '''
pygame.init()
pygame.font.init()
global canvasWidth
global canvasHeight

canvasWidth = 800
canvasHeight = 600
screen = pygame.display.set_mode((canvasWidth, canvasHeight))
background = pygame.Surface(screen.get_size())
background.fill((255,255,255))     # fill white
background = background.convert()  # jpg can not have transparency
screen.blit(background, (0,0))     # blit background on screen (overwriting all)
dead_pix = pygame.Surface([50,50])
dead_pix.fill((255,255,255))
dead_pix.convert()


##################################################################################################################################################

def newBrain():
    ''' initialize brain'''
    brain = FeedForwardNetwork()

    ''' specify layer sizes and types'''
    inLayer = LinearLayer(11, name = 'input')
    hiddenLayer1 = SigmoidLayer(7, name = 'hidden1')
    # hiddenLayer2 = SigmoidLayer(5, name = 'hidden1')
    outLayer = SoftmaxLayer(10, name = 'output')

    ''' integrate layers into brain'''
    brain.addInputModule(inLayer)
    brain.addModule(hiddenLayer1)
    # brain.addModule(hiddenLayer2)
    brain.addOutputModule(outLayer)

    ''' connect layers '''
    in_to_hidden = FullConnection(inLayer, hiddenLayer1)
    # hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
    hidden_to_out = FullConnection(hiddenLayer1, outLayer)
    brain.addConnection(in_to_hidden)
    # brain.addConnection(hidden1_to_hidden2)
    brain.addConnection(hidden_to_out)

    ''' 
        This call does some internal initialization which 
        is necessary before the net can finally be used: for 
        example, the modules are sorted topologically.
    '''
    brain.sortModules()
    
    return brain

##################################################################################################################################################

# class HPbar(pygame.sprite.Sprite):
# 	def __init__(self, sprite):
# 		pygame.sprite.Sprite.__init__(self)

# 		self.sprite = sprite
# 		# self.image = pygame.Surface((self.sprite.rect.width,7))
# 		# self.image.set_colorkey((0,0,0))
# 		# pygame.draw.rect(self.image, (0,255,0), (0,0,self.sprite.rect.width,7),1)
# 		# self.rect = self.image.get_rect()
# 		self.oldpercent = 0
# 		self.hpold = self.sprite.currentHP
# 		self.spritenumbr = self.sprite.number
# 		self.type = 'hpbar'
# 		self.font = pygame.font.SysFont('None', 10)
# 		#self.text = '%d/%d' % (self.sprite.currentHP, self.sprite.maxHP)
# 		self.sprite.hpbar = self
		

# 	def write(self, msg):
# 		self.image = self.font.render(str(msg), 1, (0,0,0))
# 		self.image.convert_alpha()
# 		self.rect = self.image.get_rect()


# 	def update(self, enemies, boxes):
# 		self.percent = self.sprite.currentHP / self.sprite.maxHP * 1.0
# 		# if self.percent < self.oldpercent:
# 		# 	pygame.draw.rect(self.image, (0,0,0), (1,1,self.sprite.rect.width - 2, 5))
# 		# 	pygame.draw.rect(self.image, (0,255,0), (1,1, int(self.sprite.rect.width * self.percent),5),0)
# 		self.text = '%d/%d' % (self.sprite.currentHP, self.sprite.maxHP)
# 		self.write(self.text)

# 		self.oldpercent = self.percent
# 		self.rect.centerx = self.sprite.rect.centerx
# 		self.rect.centery = self.sprite.rect.centery - self.sprite.rect.height / 2 - 5

class FoV(pygame.sprite.Sprite):
	def __init__(self, sprite):
		pygame.sprite.Sprite.__init__(self)

		self.sprite = sprite
		self.size = self.sprite.maximumSightDistance
		self.image = pygame.Surface([self.size, self.size])
		self.rect = self.image.get_rect()
		self.image.fill((255,245,202))

		self.pos = self.sprite.pos
		self.type = 'fov'
		self.sprite.fov = self

	def update(self, enemies, boxes):
		self.sprite.image.fill((0,0,0))
		self.pos = self.sprite.pos
		self.rect.centerx = round(self.pos[0],0)
		self.rect.centery = round(self.pos[1],0)
		self.sprite.seeAgent = 0
		self.sprite.seeEnemy = 0
		self.sprite.seeWall = 0
		self.sprite.enemy_distance = 0
		self.sprite.wall_distance = 0
		self.sprite.otherAgent_distance_x = 0
		self.sprite.otherAgent_distance_y = 0
		self.sprite.otherAgentdiffsize = 0

		enemy_direction_x = 1
		wall_direction_x = 1
		enemy_direction_y = 1
		wall_direction_y = 1
		wall_dx = 0
		wall_dy = 0
		enemy_direction = 1
		wall_direction = 1
		otherAgent_direction_x = 1
		otherAgent_direction_y = 1

		'''see other agents'''
		nearest_distance = self.sprite.maximumSightDistance
		other_agent = pygame.sprite.spritecollideany(self, boxes)
		if other_agent:
			#self.sprite.image.fill((255,162,0))
			self.sprite.seeAgent = 1
			otherAgent_dx = abs(self.sprite.pos[0] - other_agent.pos[0])
			otherAgent_dy = abs(self.sprite.pos[1] - other_agent.pos[1])

			if otherAgent_dx > 0:
				pass
			else:
				otherAgent_direction_x = -1
			if otherAgent_dy > 0:
				pass
			else:
				otherAgent_direction_y = -1

			distance_from_otherAgent = math.sqrt(otherAgent_dx ** 2 + otherAgent_dy ** 2)
			self.sprite.otherAgent_distance = distance_from_otherAgent
			self.sprite.otherAgent_distance_x = otherAgent_dx * otherAgent_direction_x
			self.sprite.otherAgent_distance_y = otherAgent_dy * otherAgent_direction_y

			self.sprite.otherAgentdiffsize = self.sprite.size - other_agent.size

		'''see enemy'''
		j = pygame.sprite.spritecollideany(self, enemies)

		if j:
			self.sprite.image.fill((255,0,0))
			self.sprite.seeEnemy = 1
			enemy_dx = abs(self.sprite.pos[0] - j.pos[0])
			enemy_dy = abs(self.sprite.pos[1] - j.pos[1])

			if enemy_dx > 0:
				pass
			else:
				enemy_direction_x = -1
			if enemy_dy > 0:
				pass
			else:
				enemy_direction_y = -1

			distance_from_enemy = math.sqrt(enemy_dx ** 2 + enemy_dy ** 2)
			self.sprite.enemy_distance = distance_from_enemy
			self.sprite.enemy_distance_x = enemy_dx * enemy_direction_x
			self.sprite.enemy_distance_y = enemy_dy * enemy_direction_y

		'''see walls'''
		if self.pos[0] + self.sprite.maximumSightDistance > self.sprite.area.width:
			self.sprite.image.fill((213, 0, 255))
			self.sprite.seeWall = 1
			wall_dx = self.sprite.area.width - self.sprite.pos[0]
			wall_direction_x = 1

		elif self.pos[0] - self.sprite.maximumSightDistance<0:
			self.sprite.image.fill((213, 0, 255))
			self.sprite.seeWall = 1
			wall_dx = self.sprite.pos[0] 
			wall_direction_x = -1

		if self.pos[1] + self.sprite.maximumSightDistance > self.sprite.area.height:
			self.sprite.image.fill((213, 0, 255))
			self.sprite.seeWall = 1
			wall_dy = self.sprite.area.height - self.sprite.pos[1]
			wall_direction_y = 1

		elif self.pos[1] - self.sprite.maximumSightDistance <0:
			self.sprite.image.fill((213, 0, 255))
			self.sprite.seeWall = 1
			wall_dy = self.sprite.pos[1] 
			wall_direction_y = -1
		
		if self.sprite.seeWall == 1:
			distance_from_wall = math.sqrt(wall_dx ** 2 + wall_dy ** 2)
			self.sprite.wall_distance = distance_from_wall
			self.sprite.wall_distance_x = wall_dx * wall_direction_x
			self.sprite.wall_distance_y = wall_dy * wall_direction_y

			
class Character(pygame.sprite.Sprite):
	characters = {}
	number = 0
	def __init__(self, area = screen, x = 1, y = 1, color = (0,0,0)):
		pygame.sprite.Sprite.__init__(self)
		self.charID = id(self)
		''' image '''
		self.size = 13.0
		self.nextsize = self.size
		self.image = pygame.Surface([self.size,self.size])
		self.rect = self.image.get_rect()
		self.image.fill(color)
		self.colorcount = 0
		self.movecount = 0
		self.type = 'character'
		self.number = Character.number
		Character.number += 1
		Character.characters[self.number] = self

		''' movement constants '''
		self.wallCollisionThreshold = 4
		self.moveSpeed = 10.0
		self.nextmoveSpeed = self.moveSpeed


		''' position '''
		self.area = area.get_rect()
		self.pos = [0.0,0.0]
		self.pos[0] = random.random() * (self.area.width - 2 * self.wallCollisionThreshold - 2 * 5)
		self.pos[1] = random.random() * (self.area.height - 2 * self.wallCollisionThreshold - 2 * 5)

		''' view constants '''
		self.maximumSightDistance = 70
		self.nextmaximumSightDistance = self.maximumSightDistance
		self.fieldOfViewSurface = pygame.Surface([self.maximumSightDistance, self.maximumSightDistance])
		self.fieldOfViewSurface.fill((255,0,0))
		self.fieldOfViewSurface.convert()
		self.fieldofViewrect = self.fieldOfViewSurface.get_rect()
		self.fieldofViewrect.clamp_ip(self.rect)
		self.fov = 0
		self.actionbar = 0
		self.hpbar = 0
		
		'''brain'''
		self.brain = 0
		self.brain_check()

		'''inputs'''
		self.seeEnemy = 0
		self.seeWall = 0
		self.enemy_distance = self.maximumSightDistance
		self.enemy_distance_x = self.maximumSightDistance
		self.enemy_distance_y = self.maximumSightDistance
		self.wall_distance = self.maximumSightDistance
		self.wall_distance_y = self.maximumSightDistance
		self.wall_distance_x = self.maximumSightDistance
		self.seeAgent = 0
		self.otherAgent_distance_x = self.maximumSightDistance
		self.otherAgent_distance_y = self.maximumSightDistance
		self.otherAgentdiffsize = 0


		'''fitness'''
		self.score = 0.0
		self.fitness = 0.0
		self.idle_damage = 1.0
		self.wall_damage = 5.0

		'''outputs'''
		self.outputs = ['U','D','L','R','UL','UR','DL','DR','dM','Attack']
		self.action = ''

		'''stats'''
		
		self.maxHP = 200.0
		self.nextmaxHP = self.maxHP
		self.currentHP = self.maxHP
		self.fitness = 0.0


	def brain_check(self):
		if isinstance(self.brain,pybrain.structure.networks.feedforward.FeedForwardNetwork) == True:
			pass #should mutate here... or not? weight training and mutation
		else:
			self.brain = newBrain()
    
	def think(self, boxes):
		inputs = self.getInputs()
		actions = self.brain.activate(inputs)
		actions = actions.tolist()
		action = self.outputs[actions.index(max(actions))]
		self.action = action
		#print self.charID, '| seeEnemy:', inputs[0], '| enemy_distance:', inputs[1], '| seeWall:', inputs[2], '| wall_distance:', inputs[3], '| currentHP:', inputs[4], '|| Score:', self.score, ' > Action:', action
		self.move(action, boxes)

	def getInputs(self):
		inputs = []
		inputs.append(self.seeEnemy)
		inputs.append(self.enemy_distance_x)
		inputs.append(self.enemy_distance_y)
		inputs.append(self.seeWall)
		inputs.append(self.wall_distance_x)
		inputs.append(self.wall_distance_y)
		inputs.append(self.seeAgent)
		inputs.append(self.otherAgent_distance_x)
		inputs.append(self.otherAgent_distance_y)
		inputs.append(self.otherAgentdiffsize)
		inputs.append(self.currentHP)
		return inputs

	def move(self,action, boxes):		
		if action == 'R':
			self.pos[0] += self.moveSpeed
			#print 'R'
			
		elif action == 'L':
			self.pos[0] += self.moveSpeed * -1
			#print 'L'
			
		elif action == 'D':
			self.pos[1] += self.moveSpeed
			#print 'U'
			
		elif action == 'U':
			self.pos[1] += self.moveSpeed * -1
			#print 'D'
		elif action == 'DL':
			self.pos[1] += self.moveSpeed
			self.pos[0] += self.moveSpeed * -1

		elif action == 'DR':
			self.pos[1] += self.moveSpeed
			self.pos[0] += self.moveSpeed

		elif action == 'UL':
			self.pos[1] += self.moveSpeed * -1
			self.pos[0] += self.moveSpeed * -1

		elif action == 'UR':
			self.pos[1] += self.moveSpeed * -1
			self.pos[0] += self.moveSpeed
		elif action == 'Attack':
			#jump
			self.pos[0] = self.otherAgent_distance_x
			self.pos[1] = self.otherAgent_distance_y

			# self.rect.centerx = self.otherAgent_distance_x
			# self.rect.centery = self.otherAgent_distance_y

			j = pygame.sprite.spritecollideany(self, boxes)
			
			if j.type == 'character':
				
				if self.size > j.size:
					
					self.colorcount = 1
					self.score += 10.0
					self.currentHP -= 1
					self.nextmoveSpeed += 5
					self.nextsize += 12
					self.nextmaxHP += 20
					self.nextmaximumSightDistance += 17	
					screen.blit(dead_pix, (j.pos[0]-j.size, j.pos[1]-j.size))
					j.kill()
					j.fov.kill()
					#j.actionbar.kill()
					#j.hpbar.kill()
					#print 'EAT!'
				elif self.size == j.size:
					if random.random() > 0.5:
						self.colorcount = 1
						self.score += 10.0
						self.currentHP -= 1
						self.nextmoveSpeed += 5
						self.nextsize += 12
						self.nextmaxHP += 20
						self.nextmaximumSightDistance += 17	
						screen.blit(dead_pix, (j.pos[0]-j.size, j.pos[1]-j.size))
						j.kill()
						j.fov.kill()
						#j.actionbar.kill()
						#j.hpbar.kill()
						#print 'EAT!'
					else:
						self.colorcount = 1
						self.score -= 5.0
						self.nextmoveSpeed -= 1
						self.nextsize -= 5
						self.nextmaxHP -= 10
						self.nextmaximumSightDistance -= 5
						self.score /= 2
						screen.blit(dead_pix, (self.pos[0]-self.size, self.pos[1]-self.size))
						self.kill()
						self.fov.kill()
						#self.actionbar.kill()
						#self.hpbar.kill()
						#print 'EATEN!!'

				else:
					self.colorcount = 1
					self.score -= 5.0
					self.nextmoveSpeed -= 1
					self.nextsize -= 5
					self.nextmaxHP -= 10
					self.nextmaximumSightDistance -= 5
					self.score /= 2
					screen.blit(dead_pix, (self.pos[0]-self.size, self.pos[1]-self.size))
					self.kill()
					self.fov.kill()
					#self.actionbar.kill()
					#self.hpbar.kill()
					#print 'EATEN!!'

		else:
			#not moving double damage
			if self.currentHP > 0:
				self.currentHP -= self.idle_damage
				#self.size -= 0.1
				#self.moveSpeed -= 0.1
			elif self.currentHP <= 0:
				self.currentHP = 0
				if self.nextsize <= 5:
					self.nextsize = 5
				else:
					self.nextsize -= 1
				if self.nextmaximumSightDistance <= self.nextsize + 10:
					self.nextmaximumSightDistance = self.nextsize + 10
				else:
					self.nextmaximumSightDistance -= 10
				if self.nextmoveSpeed <= 2:
					self.nextmoveSpeed = 2
				else:
					self.nextmoveSpeed -= 1
				self.score /= 2
				self.kill()
				self.fov.kill()
				#self.actionbar.kill()
				#self.hpbar.kill()

		'''wall management'''

		if self.pos[0] - self.size / 2< 0:
			self.pos[0] = 0
			self.pos[0] += self.size / 2
			if self.currentHP > 0:
				self.currentHP -= self.wall_damage
				#self.size -= 0.1
				#self.moveSpeed -= 0.1
			elif self.currentHP <= 0:
				self.currentHP = 0
				if self.nextsize <= 5:
					self.nextsize = 5
				else:
					self.nextsize -= 1
				if self.nextmaximumSightDistance <= self.nextsize + 10:
					self.nextmaximumSightDistance = self.nextsize + 10
				else:
					self.nextmaximumSightDistance -= 10
				if self.nextmoveSpeed <= 2:
					self.nextmoveSpeed = 2
				else:
					self.nextmoveSpeed -= 1
				self.score /= 2
				self.kill()
				self.fov.kill()
				#self.actionbar.kill()
				#self.hpbar.kill()

		elif self.pos[0] + self.size / 2 > self.area.width:
			self.pos[0] = self.area.width
			self.pos[0] -= self.size / 2
			if self.currentHP > 0:
				self.currentHP -= self.wall_damage
				#self.size -= 0.1
				#self.moveSpeed -= 0.1
			elif self.currentHP <= 0:
				self.currentHP = 0
				if self.nextsize <= 5:
					self.nextsize = 5
				else:
					self.nextsize -= 1
				if self.nextmaximumSightDistance <= self.nextsize + 10:
					self.nextmaximumSightDistance = self.nextsize + 10
				else:
					self.nextmaximumSightDistance -= 10
				if self.nextmoveSpeed <= 2:
					self.nextmoveSpeed = 2
				else:
					self.nextmoveSpeed -= 1
				self.score /= 2
				self.kill()
				self.fov.kill()
				#self.actionbar.kill()
				#self.hpbar.kill()

		if self.pos[1] - self.size / 2< 0:
			self.pos[1] = 0
			self.pos[1] += self.size / 2
			if self.currentHP > 0:
				self.currentHP -= self.wall_damage
				#self.size -= 0.1
				#self.moveSpeed -= 0.1
			elif self.currentHP <= 0:
				self.currentHP = 0
				if self.nextsize <= 5:
					self.nextsize = 5
				else:
					self.nextsize -= 1
				if self.nextmaximumSightDistance <= self.nextsize + 10:
					self.nextmaximumSightDistance = self.nextsize + 10
				else:
					self.nextmaximumSightDistance -= 10
				if self.nextmoveSpeed <= 2:
					self.nextmoveSpeed = 2
				else:
					self.nextmoveSpeed -= 1
				self.score /= 2
				self.kill()
				self.fov.kill()
				#self.actionbar.kill()
				#self.hpbar.kill()

		elif self.pos[1] + self.size / 2 > self.area.height:
			self.pos[1] = self.area.width
			self.pos[1] -= self.size / 2
			if self.currentHP > 0:
				self.currentHP -= self.wall_damage
				#self.size -= 0.1
				#self.moveSpeed -= 0.1
			elif self.currentHP <= 0:
				self.currentHP = 0
				if self.nextsize <= 5:
					self.nextsize = 5
				else:
					self.nextsize -= 1
				if self.nextmaximumSightDistance <= self.nextsize + 10:
					self.nextmaximumSightDistance = self.nextsize + 10
				else:
					self.nextmaximumSightDistance -= 10
				if self.nextmoveSpeed <= 2:
					self.nextmoveSpeed = 2
				else:
					self.nextmoveSpeed -= 1
				self.score /= 2
				self.kill()
				self.fov.kill()
				#self.actionbar.kill()
				#self.hpbar.kill()


		self.rect.centerx = round(self.pos[0],0)
		self.rect.centery = round(self.pos[1],0)
		# self.currentHP -= self.idle_damage

	def update(self, enemies, boxes):
		self.think(boxes)
		if self.colorcount != 0:
			self.movecount += 1
			if self.movecount == 4:
				self.image.fill((0,0,0))
				self.movecount = 0
		j = pygame.sprite.spritecollideany(self, enemies)
		if j:

			screen.blit(dead_pix, (j.pos[0]-self.size, j.pos[1]-self.size))
			j.kill()

			self.image.fill((0,255,0))
			self.colorcount = 1
			self.score += 1.0
			self.currentHP += 10
			self.nextmoveSpeed += 5
			self.nextsize += 10
			self.nextmaximumSightDistance += 15
		
class Enemy(pygame.sprite.Sprite):
	def __init__(self, area = screen, x = 1, y = 1, color = (0,0,255)):
		pygame.sprite.Sprite.__init__(self)
		''' image '''
		self.image = pygame.Surface([10,10])
		self.rect = self.image.get_rect()
		self.image.fill(color)

		''' movement constants '''
		self.wallCollisionThreshold = 4
		self.moveSpeed = 5

		''' view constants '''
		self.maximumSightDistance = 50
		self.fieldOfView = math.pi * 2 / 3

		''' position '''
		self.area = area.get_rect()
		self.pos = [0.0,0.0]
		self.pos[0] = random.random() * (self.area.width - 2 * self.wallCollisionThreshold - 2 * 5)
		self.pos[1] = random.random() * (self.area.height - 2 * self.wallCollisionThreshold - 2 * 5)
		self.rect.centerx = round(self.pos[0],0)
		self.rect.centery = round(self.pos[1],0)

	def move(self):
		randomMove = random.randint(1,100)
		if randomMove in range(1,25):
			self.pos[0] += self.moveSpeed
			print 'R'
			self.facing = 1
		elif randomMove in range(26,50):
			self.pos[0] += self.moveSpeed * -1
			print 'L'
			self.facing = 2
		elif randomMove in range(51,75):
			self.pos[1] += self.moveSpeed
			print 'U'
			self.facing = 3
		else:
			self.pos[1] += self.moveSpeed * -1
			print 'D'
			self.facing = 4

		if self.pos[0] < 0:
			self.pos[0] = 0
			self.pos[0] += 5
			
		elif self.pos[0] + 5 > self.area.width:
			self.pos[0] = self.area.width
			self.pos[0] -= 5

		if self.pos[1] < 0:
			self.pos[1] = 0
			self.pos[1] += 5
		elif self.pos[1] + 5 > self.area.height:
			self.pos[1] = self.area.width
			self.pos[1] -= 5

		self.rect.centerx = round(self.pos[0],0)
		self.rect.centery = round(self.pos[1],0)
		

	def update(self):
		self.move()


##################################################################################################################################################

def newPopulation(count):
	pop = []
	for i in range(count):
		newChar = Character()
		pop.append(newChar)
	return pop

def naturalSelection(population):
	matingpool = []
	maxFitness = 0.01
	r = 1.0
	'''VERSION1'''
	# for i in range(len(population)):
	# 	if population[i].fitness > maxFitness:
	# 		maxFitness = population[i].fitness

	# for i in range(len(population)):
	# 	population[i].normFitness = int(population[i].fitness/maxFitness * 100)

	# 	for j in range(population[i].normFitness):
	# 		matingpool.append(population[i])
	'''VERSION2'''
	# for i in range(len(population)):
	# 	if population[i].fitness > maxFitness:
	# 		maxFitness = population[i].fitness

	# fitlist = []
	# for i in range(len(population)):
	# 	population[i].normFitness = population[i].fitness/maxFitness 

	# fitlist = sorted(population, key = lambda x: x.normFitness)
	
	# for i in range(len(fitlist)):
	# 	if r >0:
	# 		r -= fitlist[i].normFitness
	# 		matingpool.append(fitlist[i])
	# 	else: 
	# 		break
	'''VERSION3'''
	mutationRate = 0.1
	fitlist = []
	for i in range(len(population)):
		population[i].normFitness = population[i].fitness/maxFitness 

	fitlist = sorted(population, key = lambda x: x.normFitness)
	retain = int(len(fitlist)*.3)
	parents = fitlist[:retain]
	retained = 0
	for unfit in fitlist[retain:]:
		if 0.1 > random.random():
			parents.append(unfit)
			retained += 1
	#print 'Retained:', retained
	for i in parents:
		i = mutate(i.brain, mutationRate)

	matingpool = parents


	return matingpool

def generate(population, matingpool, mutationRate):
	evolvedPop = []
	'''VERSION1'''
	# for i in range(len(population)):
	# 	male = random.randint(0, len(matingpool) - 1)
	# 	female = random.randint(0, len(matingpool) - 1)

	# 	male = matingpool[male]
	# 	female = matingpool[female]
	# 	child = crossover(male, female)
	# 	child.brain = mutate(child.brain, mutationRate)
	# 	evolvedPop.append(child)

	'''VERSION3 of naturalSelection'''
	parents_length = len(matingpool)
	desired_length = len(population) - parents_length
	children = []
	#print parents_length, desired_length
	while len(children) < desired_length:
		male = random.randint(0, len(matingpool) - 1)
		female = random.randint(0, len(matingpool) - 1)
		if male != female:
			male = matingpool[male]
			female = matingpool[female]
			for _ in range(2):
				if mutationRate > random.random():
					child = crossover(male, female)
					child.brain = mutate(child.brain, mutationRate)
					if len(children) < desired_length:
						children.append(child)

	matingpool.extend(children)

	evolvedPop = matingpool

	return evolvedPop

def crossover(male, female):
	child = Character()

	maleDNA = male.brain.params
	femaleDNA = female.brain.params

	maleDNA = maleDNA.tolist()
	femaleDNA = femaleDNA.tolist()
	'''VERSION1'''
	# split = int(len(male.brain.params) / 2)

	# maleDNA = maleDNA[:split]
	# femaleDNA = femaleDNA[split:]

	# maleDNA.extend(femaleDNA)

	# for i in range(len(child.brain.params)):
	# 	print child.brain.params[i]
	# 	child.brain.params[i] = maleDNA[i]
	# 	print child.brain.params[i], maleDNA[i]
	'''VERSION2'''
	newDNA = []
	for i in range(len(maleDNA)):
		if random.random() > 0.5:
			newDNA.append(maleDNA[i])
		else:
			newDNA.append(femaleDNA[i])	

	for i in range(len(child.brain.params)):
		child.brain.params[i] = newDNA[i]

	if random.random() > 0.5:
		child.size = male.nextsize
	else:
		child.size = female.nextsize

	if random.random() > 0.5:
		child.maximumSightDistance = male.nextmaximumSightDistance
	else:
		child.maximumSightDistance = female.nextmaximumSightDistance

	if random.random() > 0.5:
		child.moveSpeed = male.nextmoveSpeed
	else:
		child.moveSpeed = female.nextmoveSpeed

	if random.random() > 0.5:
		child.maxHP = male.nextmaxHP
	else:
		child.maxHP = female.nextmaxHP



	return child

def mutate(brain, mutationRate):
	for i in range(len(brain.params)):
		if mutationRate > random.random():
			if random.random() > 0.5:
				brain.params[i] += 0.01
			else:
				brain.params[i] -= 0.01
	return brain

def calcFitness(population):
	topFitness = []
	for i in population:
		if i.currentHP >= 0:
			i.fitness = .8 * 20 * i.score + .1 * 1  - 0.1 * .01 * (i.maxHP - i.currentHP)
		else:
			i.fitness = .8 * 20 * i.score
		topFitness.append(round(i.fitness, 2))
	topFitness = sorted(topFitness, reverse = True)
	#print topFitness

	avegenfitness = mean(topFitness)
	return topFitness, avegenfitness


##################################################################################################################################################



if __name__ == '__main__':
	clock = pygame.time.Clock()
	FPS = 60
	maxscore = 0.0
	mutationRate = 0.2
	topPrevious = [0,0,0,0,0,0,0,0,0,0]
	aveGenFitness = mean(topPrevious)

	
	pop = newPopulation(100)
	prevAlive = len(pop)
	

	boxes = pygame.sprite.LayeredUpdates()
	enemies = pygame.sprite.Group()
	fovs = pygame.sprite.Group()
	#HPbar.groups = boxes
	Character._layer = 4
	#ActionBar._layer = 3
	#HPbar._layer = 2
	FoV._layer = 1


	#boxes.add(box)
	prevRemEnemies = 20
	for gen in range(1000):
		for _ in range(0,20):
			enemies.add(Enemy())


		for char in pop:
			newFov = FoV(char)
			#newHPbar = HPbar(char)
			#newActionbar = ActionBar(char)
			boxes.add(char)
			boxes.add(newFov)
			#boxes.add(newHPbar)
			#boxes.add(newActionbar)


		for i in boxes:
			if i.type == 'character':
				boxes.move_to_front(i)
		#print boxes.sprites()

		print 'Gen: %d | Alive: %d | Rem. Enemies: %d' % (gen+1,prevAlive, prevRemEnemies), 'Top 5 Previous:', topPrevious[:5], 'Ave. Gen Fitness:', aveGenFitness

		for step in range(200):
			#print '============================'
			if len(boxes.sprites()) / 2 <= 1:
				break
			# else:
			# 	print 'Gen/Step: %d - %d | Alive: %d' % (gen+1, step+1,len(boxes.sprites())), 'Top 5 Previous:', topPrevious[:5]

			milliseconds = clock.tick(FPS)
			secondsTick = milliseconds / 1000
			playerscores = []
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					#False
					pygame.quit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						#False # user pressed ESC
						pygame.quit()
			for i in pop:
				playerscores.append(i.score)
				#playerscores.append(i.currentHP)
			maximum = max(playerscores)
			#maxscore = max(playerscores)
			pygame.display.set_caption('Top Score: %d | Step: %d' % (maximum, step))

			boxes.update(enemies, boxes)
			# fovs.update()
			#enemies.update(secondsTick)
			boxes.clear(screen, background)
			enemies.clear(screen, background)
			fovs.draw(screen)
			boxes.draw(screen)
			enemies.draw(screen)
			pygame.display.flip()
		prevAlive = len(boxes.sprites()) / 2
		prevRemEnemies = len(enemies.sprites())
		topPrevious,aveGenFitness = calcFitness(pop)

		matingpool = naturalSelection(pop)

		pop = generate(pop, matingpool, mutationRate)
		boxes.empty()
		enemies.empty()

	
	


	













