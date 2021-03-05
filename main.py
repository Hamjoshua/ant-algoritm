from itertools import groupby

# Импорт математики
import random as rn
import numpy as np
import math
from numpy.random import choice as np_choice

# Импорт pygame
import pygame

import os
import sys


IMAGE_DIR = 'source/images'
SPEED = 60


# функция для считывания входящих данных
def read_input():
    data = []
    while True:
        word = input()
        if word == '':
            break
        line = word.strip().split()
        data.append(line)
    matrix = np.array(data, float)
    print('Матрица расстояний создана')
    return matrix


def write_text(pos, text, color=(0, 0, 0)):
    font = pygame.font.SysFont('comic.ttf', 48)
    render = font.render(text, True, color)
    screen.blit(render, pos)


def load_image(name, colorkey=None):
    fullname = os.path.join('source/images', name)

    if not os.path.isfile(fullname):
        print(f"Файл с изображением '{fullname}' не найден")
        sys.exit()

    if colorkey == 0:
        image = pygame.image.load(fullname)
    else:
        image = pygame.image.load(fullname).convert()

    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey)
    else:
        image = image.convert_alpha()
    return image


class Ant(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = load_image('ant.png', colorkey=-1)
        self.rect = self.image.get_rect()
        self.text_pos = 46, 25

    def draw(self, pos, num_of_ants):
        font = pygame.font.SysFont('comic.ttf', 60)
        render = font.render(str(num_of_ants), True, (0, 0, 0))
        pos = (pos[0] - self.image.get_width() // 2, pos[1] - self.image.get_height())
        render_pos = [pos[i] + self.text_pos[i] for i in range(2)]

        screen.blit(self.image, pos)
        screen.blit(render, render_pos)


class CitiesNet:
    def __init__(self, n_inds, distances, conditions_of_ant_colony):
        self.distances = distances
        self.conditions_of_ant_colony = conditions_of_ant_colony
        self.cur_iter = 0
        self.coords = dict()
        self.n_inds = n_inds
        center = screen.get_width() // 2, screen.get_height() // 2
        height = 160

        # нахождение основных городов
        angle = ((n_inds - 2) / n_inds) * 180
        const_rad_angle = rad_angle = angle * (math.pi / 180)
        self.coords[0] = center[0], center[1] - height
        for ind in range(1, n_inds):
            x1, y1 = center[0] + (self.coords[0][0] - center[0]) * math.cos(rad_angle) - \
                     (self.coords[0][1] - center[1]) * math.sin(rad_angle), \
                     center[1] + (self.coords[0][0] - center[0]) * math.sin(rad_angle) + \
                     (self.coords[0][1] - center[1]) * math.cos(rad_angle)

            rad_angle += const_rad_angle
            self.coords[ind] = x1, y1

        # нахождение дорог между городами
        self.trips = []
        for y in range(len(self.distances)):
            for x in range(len(self.distances[0])):
                if self.distances[y][x] != np.inf:
                    self.trips.append((y, x))

    def draw_net(self):
        # окрашивание тропинок между городами и основных дорог
        for y in range(len(self.distances)):
            for x in range(len(self.distances[0])):
                if self.distances[y][x] != np.inf:
                    pherm_intencity = self.conditions_of_ant_colony[self.cur_iter]['pheromone'][y][x]
                    color = (min(255, pherm_intencity * 255), 0, 0)
                    pygame.draw.line(screen, color, self.coords[y], self.coords[x], 10)

        # нумерация города
        list_of_coords = [self.coords[i] for i in range(len(self.coords))]
        for i in range(len(list_of_coords)):
            write_text(list_of_coords[i], str(i + 1), color=(0, 255, 0))

    def draw_ant(self, timer):
        ant = Ant()

        current_path_index = timer // SPEED
        coef = (timer % SPEED) / SPEED

        if self.conditions_of_ant_colony[cur_iter]['all_paths']:
            sorted_all_paths = sorted(self.conditions_of_ant_colony[cur_iter]['all_paths'],
                                      key=lambda x: x[1])

            current_inds = [sorted_all_paths[i][0][current_path_index] for i in
                             range(len(sorted_all_paths))]
            print(timer, current_path_index, coef, SPEED * self.n_inds)
            print(current_inds)

            print('-' * 5)
            for ind in current_inds:
                first_ind, second_ind = self.coords[ind[0]], self.coords[ind[1]]
                if second_ind[0] > first_ind[0]:
                    x = first_ind[0] + round(((second_ind[0] - first_ind[0]) * coef))
                else:
                    x = first_ind[0] - round(((first_ind[0] - second_ind[0]) * coef))
                if second_ind[1] > first_ind[1]:
                    y = first_ind[1] + round(((second_ind[1] - first_ind[1]) * coef))
                else:
                    y = first_ind[1] - round(((first_ind[1] - second_ind[1]) * coef))
                pos = (x, y)
                ant.draw(pos, current_inds.count(ind))

    def next_iter(self):
        self.cur_iter += 1


class AntColony(object):
    """
    Находит кротчайший путь для решения задачи коммивояжера.

    Attributes
    ----------
    distances : np.array
    Матрица расстояний вида
      1  2  3  -  список городов
    1 0  x  y  -  расстояния от города до другого города. Между собой город не имеет расстояний
    2 x  0  z
    3 y  z  0

    """

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        # избавляемся от нулей в матрице
        i = 0
        j = 0
        while i < distances.shape[0]:
            while j < distances.shape[1]:
                if distances[i][j] == 0:
                    distances[i][j] = np.inf
                    i += 1
                    j += 1
                else:
                    j += 1
                    continue
            i += 1
        self.distances = distances  # матрица растояний
        self.pheromone = np.ones(self.distances.shape) / len(distances)  # матрица феромонов
        self.all_inds = range(len(distances))  # список городов
        self.n_ants = n_ants  # колличество муравьев
        self.n_best_ants = n_best  # колличество элитных муравьев
        self.n_iterations = n_iterations  # колличество итераций
        self.decay = decay  # испарения феромона
        self.alpha = alpha
        self.beta = beta

    def run(self, debug=False):
        """
        Функция возвращает короткий путь. На каждой итерации расчитывается список всех путей и распро
        страняется феромон в зависимости от условий. В конце итерации феромон испаряется по параметру
        decay.

        Parameters
        ----------
        debug : bool
            включение отладки в консоль
        """

        conditions_of_ant_colony = []
        conditions_of_ant_colony.append({'all_paths': None,
                                         'pheromone': np.copy(self.pheromone)})
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best_ants, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
            if debug:
                self.debug_condition(i)
            conditions_of_ant_colony.append({'all_paths': all_paths,
                                             'pheromone': np.copy(self.pheromone)})
        return all_time_shortest_path, conditions_of_ant_colony

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        """

        Parameters
        ----------
        :param all_paths: list
            список всех путей в виде кортежей
        :param n_best: int
            количество элитных муравьев
        :param shortest_path: tuple(int)
            самый короткий путь
        :return: None
        """

        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        """
        Считает роасстояние для пути из городов

        :param path: tuple(int)
            путь
        :return: total_dist : int
            общее расстояние для пути
        """

        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        """
        Генерериует список всех путей

        :return: list(tuple)
            список всех путей
        """

        all_paths = []
        for ant in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        """
        Генерирует путь с города start

        :param start: int
            Город, с которого начинается путь
        :return: list(int)
            Путь-список, состоящий из посещенных городов
        """

        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # возвращаемся к тому, с чего начали
        return path

    def pick_move(self, pheromone, dist, visited):
        """
        Выбирает город, в который пойдет муравей. На выбор влияет интенсивность феромона для узла
        городов

        :param pheromone: np.array([x, y, z])
            срез по иксу матрицы феромонов
        :param dist: int
            Длина пути
        :param visited:
            Табу или список запрещенных городов. Интенсивность феромонов для табу равна нулю, чтобы
            муравей не вернулся обратно
        :return: int
            Номер выбранного города
        """

        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0  # посещенные города не должны посещаться снова
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def __str__(self):
        return f"Матрица расстояний: \n{self.distances}\n" \
               f"Матрица феромонов: \n{self.pheromone}\n" \
               f"Список городов: {self.all_inds}\n" \
               f"Кол-во муравьев/эл. муравьев: {self.n_ants, self.n_best_ants}\n" \
               f"Кол-во итераций: {self.n_iterations}\n" \
               f"Испарение феромона: {self.decay}\n" \
               f"Альфа и бета: {self.alpha, self.beta}"

    def debug_condition(self, n_iteration):
        print(f'------Шаг-{n_iteration + 1}.')
        for y in range(len(self.pheromone)):
            for x in range(len(self.pheromone[0])):
                if x != y:
                    print(f'Узел {y + 1}-{x + 1}: {self.pheromone[y][x]}')


if __name__ == '__main__':
    distance = read_input()
    ant_colony = AntColony(distances=distance,
                           n_ants=len(distance) * 2,
                           n_best=5,
                           n_iterations=len(distance) * 4,
                           decay=0.95,
                           alpha=1,
                           beta=1)
    print(ant_colony)
    shortest_path, conditions_of_ant_colony = ant_colony.run()
    print(f"Кратчайший путь: {shortest_path}")

    pygame.init()
    pygame.display.set_caption('Ant algorithm')
    WIDTH, HEIGHT = SIZE = 800, 600
    screen = pygame.display.set_mode(SIZE)
    fps = 30
    clock = pygame.time.Clock()

    n_iterations = ant_colony.n_iterations
    cur_iter = 0
    timer = 0
    n_ants = ant_colony.n_ants

    city_net = CitiesNet(n_inds=len(ant_colony.all_inds), distances=ant_colony.distances,
                         conditions_of_ant_colony=conditions_of_ant_colony)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        screen.fill('White')

        if cur_iter >= n_iterations:
            write_text((100, 100), "Анимация закончилась")
        elif timer == len(ant_colony.all_inds) * SPEED and cur_iter < n_iterations:
            timer = 0
            city_net.next_iter()
            cur_iter += 1

        write_text((0, 0), str(cur_iter))

        city_net.draw_net()
        city_net.draw_ant(timer)
        clock.tick(fps)
        pygame.display.flip()

        timer += 1

input()
