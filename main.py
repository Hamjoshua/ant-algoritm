from itertools import groupby

# Импорт математики
import random as rn
import numpy as np
import math

from PyQt5.QtWidgets import QApplication, QMainWindow
from numpy.random import choice as np_choice

# Импорт pygame
import pygame

import os
import sys

IMAGE_DIR = 'source/images'
DURATION = 60


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


def write_text(pos, text, screen, color=(0, 0, 0), centered=False):
    font = pygame.font.SysFont('comic.ttf', 48)
    render = font.render(text, True, color)
    if centered:
        screen.blit(render, (pos[0] - render.get_width() // 2, pos[1]))
    else:
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


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


class Ui_interface(QtWidgets.QMainWindow):
    def __init__(self, game):
        super(Ui_interface, self).__init__()
        self.setupUi()
        self.init_pygame(game)
        self.is_pause = False
        self.cur_iter = 0
        self.setup_buttons()

    def setupUi(self):
        interface = self
        interface.setObjectName("interface")
        interface.resize(404, 300)
        interface.setObjectName("interface")
        interface.resize(404, 336)
        self.centralwidget = QtWidgets.QWidget(interface)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setRowCount(6)
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setDefaultSectionSize(39)
        self.gridLayout.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.animation_manager = QtWidgets.QWidget(self.centralwidget)
        self.animation_manager.setEnabled(False)
        self.animation_manager.setObjectName("animation_manager")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.animation_manager)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.play = QtWidgets.QPushButton(self.animation_manager)
        self.play.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(IMAGE_DIR, "play-button.png")), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.play.setIcon(icon)
        self.play.setIconSize(QtCore.QSize(32, 32))
        self.play.setAutoDefault(False)
        self.play.setDefault(False)
        self.play.setFlat(True)
        self.play.setObjectName("play")
        self.gridLayout_2.addWidget(self.play, 0, 2, 1, 1)
        self.prev = QtWidgets.QPushButton(self.animation_manager)
        self.prev.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(IMAGE_DIR, "prev.png")), QtGui.QIcon.Normal,
                        QtGui.QIcon.Off)
        self.prev.setIcon(icon1)
        self.prev.setIconSize(QtCore.QSize(32, 32))
        self.prev.setFlat(True)
        self.prev.setObjectName("prev")
        self.gridLayout_2.addWidget(self.prev, 0, 0, 1, 1)
        self.pause = QtWidgets.QPushButton(self.animation_manager)
        self.pause.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join(IMAGE_DIR, "pause.png")), QtGui.QIcon.Normal,
                        QtGui.QIcon.Off)
        self.pause.setIcon(icon2)
        self.pause.setIconSize(QtCore.QSize(32, 32))
        self.pause.setFlat(True)
        self.pause.setObjectName("pause")
        self.gridLayout_2.addWidget(self.pause, 0, 1, 1, 1)
        self.next = QtWidgets.QPushButton(self.animation_manager)
        self.next.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join(IMAGE_DIR, "next.png")), QtGui.QIcon.Normal,
                        QtGui.QIcon.Off)
        self.next.setIcon(icon3)
        self.next.setIconSize(QtCore.QSize(32, 32))
        self.next.setFlat(True)
        self.next.setObjectName("next")
        self.gridLayout_2.addWidget(self.next, 0, 3, 1, 1)
        self.cadre_label = QtWidgets.QLabel(self.animation_manager)
        self.cadre_label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.cadre_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.cadre_label.setObjectName("cadre_label")
        self.gridLayout_2.addWidget(self.cadre_label, 1, 1, 1, 1)
        self.cadre_box = QtWidgets.QSpinBox(self.animation_manager)
        self.cadre_box.setMaximum(0)
        self.cadre_box.setObjectName("cadre_box")
        self.gridLayout_2.addWidget(self.cadre_box, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.animation_manager, 2, 0, 1, 2)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.alpha_label = QtWidgets.QLabel(self.centralwidget)
        self.alpha_label.setObjectName("alpha_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.alpha_label)
        self.beta_label = QtWidgets.QLabel(self.centralwidget)
        self.beta_label.setObjectName("beta_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.beta_label)
        self.decay_label = QtWidgets.QLabel(self.centralwidget)
        self.decay_label.setObjectName("decay_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.decay_label)
        self.best_label = QtWidgets.QLabel(self.centralwidget)
        self.best_label.setObjectName("best_label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.best_label)
        self.alpha_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.alpha_box.setMaximum(1.0)
        self.alpha_box.setSingleStep(0.05)
        self.alpha_box.setProperty("value", 1.0)
        self.alpha_box.setObjectName("alpha_box")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.alpha_box)
        self.beta_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.beta_box.setSingleStep(0.05)
        self.beta_box.setProperty("value", 1.0)
        self.beta_box.setObjectName("beta_box")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.beta_box)
        self.decay_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.decay_box.setMaximum(1.0)
        self.decay_box.setSingleStep(0.05)
        self.decay_box.setProperty("value", 0.9)
        self.decay_box.setObjectName("decay_box")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.decay_box)
        self.best_box = QtWidgets.QSpinBox(self.centralwidget)
        self.best_box.setMaximum(10)
        self.best_box.setProperty("value", 5)
        self.best_box.setObjectName("best_box")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.best_box)
        self.gridLayout.addLayout(self.formLayout, 0, 1, 1, 1)
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 1, 0, 1, 2)
        interface.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(interface)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 404, 21))
        self.menubar.setObjectName("menubar")
        interface.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(interface)
        self.statusbar.setObjectName("statusbar")
        interface.setStatusBar(self.statusbar)
        self.duration_label = QtWidgets.QLabel(self.animation_manager)
        self.duration_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.duration_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.duration_label.setObjectName("duration_label")
        self.gridLayout_2.addWidget(self.duration_label, 2, 0, 1, 2)
        self.duration_box = QtWidgets.QSpinBox(self.animation_manager)
        self.duration_box.setMaximum(100)
        self.duration_box.setObjectName("duration_box")
        self.gridLayout_2.addWidget(self.duration_box, 2, 2, 1, 1)

        self.retranslateUi(interface)
        QtCore.QMetaObject.connectSlotsByName(interface)

        self.is_ready = False

    def init_pygame(self, game):
        self.game = game
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(0)

    def pygame_loop(self):
        if self.game.is_ready:
            if self.game.loop(self):
                self.close()
            self.cadre_box.setValue(self.game.timer)

    def retranslateUi(self, interface):
        _translate = QtCore.QCoreApplication.translate
        interface.setWindowTitle(_translate("interface", "interface"))
        self.cadre_label.setText(_translate("interface", "Текущий кадр:"))
        self.duration_label.setText(_translate("interface", "Продолжительность анимации:"))
        self.alpha_label.setText(_translate("interface", "alpha"))
        self.beta_label.setText(_translate("interface", "beta"))
        self.decay_label.setText(_translate("interface", "decay"))
        self.best_label.setText(_translate("interface", "best"))
        self.start.setText(_translate("interface", "Начать анимацию"))

    def setup_buttons(self):
        self.start.clicked.connect(self.start_animation)
        self.pause.clicked.connect(self.set_pause)
        self.play.clicked.connect(self.disable_pause)
        self.next.clicked.connect(self.next_iter)
        self.prev.clicked.connect(self.prev_iter)
        self.cadre_box.valueChanged.connect(self.change_cadre)
        self.duration_box.valueChanged.connect(self.change_duration)

    def set_pause(self):
        self.is_pause = True

    def disable_pause(self):
        self.is_pause = False

    def next_iter(self):
        if self.game.cur_iter + 1 <= self.game.n_iterations:
            self.game.cur_iter += 1
            self.game.timer = 0

    def prev_iter(self):
        if self.game.cur_iter - 1 >= 0:
            self.game.cur_iter -= 1
            self.game.timer = 0

    def change_cadre(self):
        cadre = int(self.cadre_box.value())
        self.game.timer = cadre

    def change_duration(self):
        global DURATION
        duration = int(self.duration_box.value())
        DURATION = duration
        self.cadre_box.setMaximum(self.game.n_inds * DURATION)
        self.game.timer = min(self.game.timer, self.game.n_inds * DURATION)

    def start_animation(self):
        try:
            vals_dicty = dict()

            vals_dicty['alpha'] = float(self.alpha_box.value())
            vals_dicty['beta'] = float(self.beta_box.value())
            vals_dicty['decay'] = float(self.decay_box.value())
            vals_dicty['n_best'] = int(self.best_box.value())

            distance = []

            for row in range(self.tableWidget.rowCount()):
                distance.append(list())
                for column in range(self.tableWidget.columnCount()):
                    item = self.tableWidget.item(row, column)
                    if item is None:
                        break
                    item_val = int(item.text())
                    if row == column:
                        if item_val != 0:
                            raise ValueError(
                                'Расстояние города между собой должно быть равно нулю')
                    distance[row].append(item_val)
            distance = [i for i in distance if i]
            vals_dicty['distances'] = np.array(distance, float)
            vals_dicty['n_ants'] = len(distance) * 2
            vals_dicty['n_iterations'] = len(distance) * 4

            self.game.game_init(vals_dicty)
            self.animation_manager.setEnabled(True)
            self.cadre_box.setMaximum(self.game.n_inds * DURATION)
            self.duration_box.setValue(DURATION)
        except Exception as e:
            raise e


class Ant(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = load_image('ant.png', colorkey=-1)
        self.rect = self.image.get_rect()
        self.text_pos = 46, 25

    def draw(self, pos, num_of_ants, screen):
        font = pygame.font.SysFont('comic.ttf', 60)
        render = font.render(str(num_of_ants), True, (0, 0, 0))
        pos = (pos[0] - self.image.get_width() // 2, pos[1] - self.image.get_height())
        render_pos = [pos[i] + self.text_pos[i] for i in range(2)]

        screen.blit(self.image, pos)
        screen.blit(render, render_pos)


class CitiesNet:
    def __init__(self):
        self.is_ready = False
        pygame.init()
        pygame.display.set_caption('Ant algorithm')
        self.WIDTH, self.HEIGHT = self.SIZE = 800, 600
        self.screen = pygame.display.set_mode(self.SIZE)
        self.fps = 30
        self.clock = pygame.time.Clock()

    def draw_net(self):
        # окрашивание тропинок между городами и основных дорог
        for y in range(len(self.distances)):
            for x in range(len(self.distances[0])):
                if self.distances[y][x] != np.inf:
                    pherm_intencity = self.conditions_of_ant_colony[self.cur_iter]['pheromone'][y][x]
                    color = (min(255, pherm_intencity * 255), 0, 0)
                    pygame.draw.line(self.screen, color, self.coords[y], self.coords[x], 10)

        # нумерация города
        list_of_coords = [self.coords[i] for i in range(len(self.coords))]
        for i in range(len(list_of_coords)):
            write_text(list_of_coords[i], str(i + 1), color=(0, 255, 0), screen=self.screen)

    def draw_ant(self, timer):
        ant = Ant()

        current_path_index = timer // DURATION
        coef = (timer % DURATION) / DURATION

        if self.conditions_of_ant_colony[self.cur_iter]['all_paths']:
            sorted_all_paths = sorted(self.conditions_of_ant_colony[self.cur_iter]['all_paths'],
                                      key=lambda x: x[1])

            current_inds = [sorted_all_paths[i][0][current_path_index] for i in
                            range(len(sorted_all_paths))]
            print(timer, current_path_index, coef, DURATION * self.n_inds)
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
                ant.draw(pos, current_inds.count(ind), self.screen)

    def game_init(self, values):
        ant_colony = AntColony(**values)
        print(ant_colony)
        shortest_path, conditions_of_ant_colony = ant_colony.run()
        print(f"Кратчайший путь: {shortest_path}")

        self.n_iterations = ant_colony.n_iterations
        self.n_inds = len(ant_colony.all_inds)
        self.conditions_of_ant_colony = conditions_of_ant_colony
        self.cur_iter = 0
        self.timer = 0
        self.n_ants = ant_colony.n_ants
        self.distances = ant_colony.distances
        self.coords = dict()

        center = self.screen.get_width() // 2, self.screen.get_height() // 2
        height = 160

        # нахождение основных городов
        angle = ((self.n_inds - 2) / self.n_inds) * 180
        const_rad_angle = rad_angle = angle * (math.pi / 180)
        self.coords[0] = center[0], center[1] - height
        for ind in range(1, self.n_inds):
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

        self.is_ready = True

    def loop(self, window):
        if self.is_ready:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True

            self.screen.fill('White')

            if self.cur_iter >= self.n_iterations:
                write_text((self.screen.get_width() // 2, 500), "Анимация закончилась", self.screen,
                           centered=True)
            elif self.timer == self.n_inds * DURATION and self.cur_iter < self.n_iterations:
                self.timer = 0
                self.cur_iter += 1
            self.draw_net()
            self.draw_ant(min(self.timer, (self.n_inds * DURATION) - 1))

            write_text((0, 0), str(self.cur_iter), self.screen)

            self.clock.tick(self.fps)
            pygame.display.flip()
            if not window.is_pause:
                self.timer += 1
            return False


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


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    game = CitiesNet()
    app = QApplication(sys.argv)
    ex = Ui_interface(game)
    ex.show()
    sys.excepthook = except_hook
    result = app.exec()
    sys.exit(result)

input()
