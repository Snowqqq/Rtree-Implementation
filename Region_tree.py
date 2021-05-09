import math
import pandas as pd
import numpy as np
import sys


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def perimeter(self):
        return 2 * (abs(self.x2 - self.x1) + abs(self.y2 - self.y1))

    # judge whether two rect are overlap
    def is_overlap(self, rect):
        if self.y1 > rect.y2 or self.y2 < rect.y1 or self.x1 > rect.x2 or self.x2 < rect.y1:
            return False
        return True

    # judge whether the rect contains another tec
    def contain_rect(self, rect):
        return self.x1 < rect.x1 and self.y1 < rect.y1 and self.x2 > rect.x2 and self.y2 > rect.y2

    # judge whether the rect contains the point
    def has_point(self, point):
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def __str__(self):
        return "Rect: ({}, {}), ({}, {})".format(self.x1, self.y1, self.x2, self.y2)


class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        return "Point #{}: ({}, {})".format(self.id, self.x, self.y)


def sequential_query(points, query):
    result = 0
    for point in points:
        if query.x1 <= point.x <= query.x2 and query.y1 <= point.y <= query.y2:
            result = result + 1
    return result


class Node(object):
    def __init__(self, d, n, height):
        self.d = d
        self.n = n
        self.id = 0
        self.height = height
        # for internal nodes
        self.child_nodes = []
        # for leaf nodespyth
        self.data_points = []
        self.parent_node = None
        self.MBR = Rect(-1, -1, -1, -1)

    def add_point(self, point):
        # update in the right position to keep the list ordered
        self.add_points([point])
        pass

    def add_points(self, points):
        self.data_points += points
        # update MBR
        self.update_MBR()
        pass

    # calculate perimeter increment when add a point
    def perimeter_increase_with_point(self, point):
        x1 = point.x if point.x < self.MBR.x1 else self.MBR.x1
        y1 = point.y if point.y < self.MBR.y1 else self.MBR.y1
        x2 = point.x if point.x > self.MBR.x2 else self.MBR.x2
        y2 = point.y if point.y > self.MBR.y2 else self.MBR.y2
        return Rect(x1, y1, x2, y2).perimeter() - self.perimeter()

    def perimeter(self):
        # only calculate the half perimeter here
        return self.MBR.perimeter()

    # according to the node is leaf or non-leaf, and judge whether to underflow
    def is_underflow(self):
        return (self.is_leaf() and len(self.data_points) < math.ceil(self.n / 2)) or \
               (not self.is_leaf() and len(self.child_nodes) < math.ceil(self.d / 2))

    def is_overflow(self):
        return (self.is_leaf() and len(self.data_points) > self.n) or \
               (not self.is_leaf() and len(self.child_nodes) > self.d)

    def is_root(self):
        return self.parent_node is None

    def is_leaf(self):
        return len(self.child_nodes) == 0

    def add_child_node(self, node):
        self.add_child_nodes([node])
        pass

    def add_child_nodes(self, nodes):
        for node in nodes:
            node.parent_node = self
            self.child_nodes.append(node)
        self.update_MBR()
        pass

    # update MBR for leaf and non-leaf situations
    def update_MBR(self):
        if self.is_leaf():
            self.MBR.x1 = min([point.x for point in self.data_points])
            self.MBR.x2 = max([point.x for point in self.data_points])
            self.MBR.y1 = min([point.y for point in self.data_points])
            self.MBR.y2 = max([point.y for point in self.data_points])
        else:
            self.MBR.x1 = min([child.MBR.x1 for child in self.child_nodes])
            self.MBR.x2 = max([child.MBR.x2 for child in self.child_nodes])
            self.MBR.y1 = min([child.MBR.y1 for child in self.child_nodes])
            self.MBR.y2 = max([child.MBR.y2 for child in self.child_nodes])
        if self.parent_node and not self.parent_node.MBR.contain_rect(self.MBR):
            self.parent_node.update_MBR()
        pass

    # Get perimeter of an MBR formed by a list of data points: by finding four points in MBR
    @staticmethod
    def get_points_MBR_perimeter(points):
        x1 = min([point.x for point in points])
        x2 = max([point.x for point in points])
        y1 = min([point.y for point in points])
        y2 = max([point.y for point in points])
        return Rect(x1, y1, x2, y2).perimeter()

    @staticmethod
    def get_nodes_MBR_perimeter(nodes):
        x1 = min([node.MBR.x1 for node in nodes])
        x2 = max([node.MBR.x2 for node in nodes])
        y1 = min([node.MBR.y1 for node in nodes])
        y2 = max([node.MBR.y2 for node in nodes])
        return Rect(x1, y1, x2, y2).perimeter()


class RegionTree:
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.root = Node(self.d, self.n, height = 1)


    def insert_point(self, point, cur_node = None):
        # init U as node
        # print("{} is leaf: {}".format(self.root, self.root.is_leaf()))
        if cur_node is None:
            cur_node = self.root

            # print("{} is leaf: {}".format(cur_node, cur_node.is_leaf()))
        # Insertion logic start
        if cur_node.is_leaf():
            cur_node.add_point(point)
            # handle overflow
            if cur_node.is_overflow():
                self.handle_overflow(cur_node)
        else:
            chosen_child = self.choose_best_child(cur_node, point)
            self.insert_point(point, cur_node=chosen_child)

    # Find a suitable one to expand:
    # the point index = 0 or the perimeter increment be the largest
    @staticmethod
    def choose_best_child(node, point):
        best_child = None
        best_perimeter = 0
        # Scan the child nodes
        for item in node.child_nodes:
            if node.child_nodes.index(item) == 0 or best_perimeter > item.perimeter_increase_with_point(point):
                best_child = item
                best_perimeter = item.perimeter_increase_with_point(point)
        return best_child

    # solve the overflow situation
    def handle_overflow(self, node):
        node, new_node = self.split_leaf_node(node) if node.is_leaf() else self.split_internal_node(node)
        if self.root is node:
            self.root = Node(self.d, self.n, height = node.height + 1)
            self.root.add_child_nodes([node, new_node])
        else:
            node.parent_node.add_child_node(new_node)
            if node.parent_node.is_overflow():
                self.handle_overflow(node.parent_node)

    # split leaf node, find max mbr and update set1 and set2
    def split_leaf_node(self, node):
        m = len(node.data_points)
        best_perimeter = -1
        best_set_1 = []
        best_set_2 = []
        # Run x axis, and sort all poins by x
        all_point_sorted_by_x = sorted(node.data_points, key=lambda point: point.x)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_point_1 = all_point_sorted_by_x[:i]
            list_point_2 = all_point_sorted_by_x[i:]
            temp_sum_perimeter = Node.get_points_MBR_perimeter(list_point_1) \
                                 + Node.get_points_MBR_perimeter(list_point_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_point_1
                best_set_2 = list_point_2
        # Run y axis, and sort all poins by y
        all_point_sorted_by_y = sorted(node.data_points, key=lambda point: point.y)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_point_1 = all_point_sorted_by_y[:i]
            list_point_2 = all_point_sorted_by_y[i:]
            temp_sum_perimeter = Node.get_points_MBR_perimeter(list_point_1) \
                                 + Node.get_points_MBR_perimeter(list_point_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_point_1
                best_set_2 = list_point_2
        node.data_points = best_set_1
        node.update_MBR()
        new_node = Node(self.d, self.n, height = node.height)
        new_node.add_points(best_set_2)
        return node, new_node

    # split non-leaf node
    def split_internal_node(self, node):
        m = len(node.child_nodes)
        best_perimeter = -1
        best_set_1 = []
        best_set_2 = []
        # Run x axis
        all_node_sorted_by_x = sorted(node.child_nodes, key=lambda child: child.MBR.x1)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_node_1 = all_node_sorted_by_x[:i]
            list_node_2 = all_node_sorted_by_x[i:]
            temp_sum_perimeter = Node.get_nodes_MBR_perimeter(list_node_1) \
                                 + Node.get_nodes_MBR_perimeter(list_node_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_node_1
                best_set_2 = list_node_2
                # Run y axis
        all_node_sorted_by_y = sorted(node.child_nodes, key=lambda child: child.MBR.y1)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_node_1 = all_node_sorted_by_y[:i]
            list_node_2 = all_node_sorted_by_y[i:]
            temp_sum_perimeter = Node.get_nodes_MBR_perimeter(list_node_1) \
                                 + Node.get_nodes_MBR_perimeter(list_node_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_node_1
                best_set_2 = list_node_2
        node.child_nodes = best_set_1
        node.update_MBR()
        new_node = Node(self.d, self.n, height = node.height)
        new_node.add_child_nodes(best_set_2)
        return node, new_node

    # Take in a Rect and return number of data point that is covered by the R tree.
    def region_query(self, rect, node=None):
        # initiate with root
        if node is None:
            node = self.root

        if node.is_leaf():
            # print("get here")
            count = 0
            for point in node.data_points:
                if rect.has_point(point):
                    count += 1
            return count
        else:
            # print([child.MBR for child in node.child_nodes])
            total = 0
            for child in node.child_nodes:
                # print("{} and {} is overlapped {}".format(rect, child.MBR, rect.is_overlap(child.MBR)))
                if rect.is_overlap(child.MBR):
                    total += self.region_query(rect, child)
            return total

# create an R-tree by user-give d, n, total用来调试取前几行数据
def createTree(d, n, pointsList, total = 0):
    Rtree = RegionTree(d, n)
    if total != 0:
        num = total
    else:
        num = len(pointsList)
    # insert point to create rtree
    for i in range(num):
       Rtree.insert_point(Point(pointsList[i][0], pointsList[i][1], pointsList[i][2]))
    return Rtree

# count the numbers of non-leaf and leaf nodes
def classify_nodes(Rtree):
   stack = [Rtree.root]
   non_leaf = []; leaf = []
   while stack:
    node = stack.pop(0)
    if node.is_leaf():
        leaf.append(node)
    else:
        stack += node.child_nodes
        non_leaf.append(node)
   return non_leaf, leaf

# calculate  space utilisation for leaf nodes
def count_spaceUtilisation(leaf):
    space_utilisation = [0, 0, 0, 0]
    for node in leaf:
      utility = len(node.data_points) / node.d
      if utility < 0.25:
        space_utilisation[0] += 1
      elif utility < 0.5:
        space_utilisation[1] += 1
      elif utility < 0.75:
        space_utilisation[2] += 1
      else:
        space_utilisation[3] += 1
    return space_utilisation


# calculate sub-tree overlapping among non-leaf nodes
def MBRoverlap_count(Rtree):
    cntOverlap = 0
    stack = [Rtree.root]
    while stack:
        node = stack.pop(0)
        if not node.is_leaf():
            stack += node.child_nodes
            child = node.child_nodes
            for i in range(len(child)):
                for j in range(i + 1, len(child)):
                    if child[i].MBR.is_overlap(child[j].MBR):
                        cntOverlap += 1
                        break
    return cntOverlap


# Task 3
# calculate the lower bound distance for any point in R to q
def calMinDist(R, q):
    if R.has_point(q):
        minDist = 0
    else:
        if (R.x1 <= q.x and R.x2 >= q.x) or (R.y1 <= q.y and R.y2 >= q.y):
            minDist = min(abs(R.x1 - q.x), abs(R.x2 - q.x), abs(R.y1 - q.y), abs(R.y2 - q.y))
        else:
            minDist = min(calDistance(R.x1, q.x, R.y1, q.y), calDistance(R.x1, q.x, R.y2, q.y),
                          calDistance(R.x2, q.x, R.y1, q.y), calDistance(R.x2, q.x, R.y2, q.y))
    return minDist

# calculate an upper bound of distance from q to R
def calMinMaxDist(R, q):
    d1 = calDistance(R.x1, q.x, R.y1, q.y)
    d2 = calDistance(R.x1, q.x, R.y2, q.y)
    d3 = calDistance(R.x2, q.x, R.y1, q.y)
    d4 = calDistance(R.x2, q.x, R.y2, q.y)
    max1 = max(d1, d2)
    max2 = max(d1, d3)
    max3 = max(d2, d4)
    max4 = max(d3, d4)
    return min(max1, max2, max3, max4)

# calculate two points distance
def calDistance(x1, x2, y1, y2):
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

# find the nearest neighbour of a given query point based on the R-tree
def search(tupleList, query):
    global distance
    global results
    global cntNodes
    global cntPoints
    global cntMBRs
    global nodes
    global id
    cntNodes += 1
    if tupleList[0][1].is_leaf():
        node = tupleList[0][1]
        # remove the first element in the tuple list
        del tupleList[0]
        for point in node.data_points:
            new_dis = calDistance(point.x, query.x, point.y, query.y)
            cntPoints += 1
            if new_dis <= distance:
                results.clear()
                results.append(point)
                if new_dis < distance:
                    distance = new_dis
        # print('id{}.end'.format(id))
        return

        # for non-leaf nodes
    else:
        node = tupleList[0][1]
        # store minDist and node in ABL and sort it by minDist
        ABL = [(calMinDist(child.MBR, query), child) for child in node.child_nodes]
        ABL = sorted(ABL, key=lambda t: t[0])
        ABL_new = []
        del tupleList[0]
        while (len(ABL) > 1):
            MBR0 = ABL[0]
            MBR1 = ABL[1]
            # pruning rule 1: if there exists another R’, MINDIST(q, R) > MINMAXDIST(q, R’): an MBR R is discarded
            if MBR0[0] > calMinMaxDist(MBR1[1].MBR, query):
                ABL.pop(0)
                cntMBRs += 1  # ;print('id:{},cut1'.format(id))
            elif MBR1[0] > calMinMaxDist(MBR0[1].MBR, query):
                ABL.pop(1)
                cntMBRs += 1  # ;print('id:{},cut2'.format(id))
            else:
                # pruning rule 3: if an object o is found the MINDIST(q, R) > Actual-Dist(q, o): an MBR R is discarded
                # print('id:{}:'.format(id))
                if node not in nodes.keys():
                    nodes[node] = {}

                if MBR0 in nodes[node].keys():
                    p0 = nodes[node][MBR0]
                else:
                    search([MBR0], query)
                    p0 = results[0]
                    nodes[node][MBR0] = p0
                    id += 1
                if MBR1 in nodes[node].keys():
                    p1 = nodes[node][MBR1]
                else:
                    search([MBR1], query)
                    p1 = results[0]
                    nodes[node][MBR1] = p1

                # pruning rule 2:
                # if there exists an R the Actual-Dist(q, o) > MINMAXDIST(q, R): an object o is discarded
                if MBR1[0] > calDistance(p0.x, query.x, p0.y, query.y):
                    ABL.pop(1)
                    cntMBRs += 1
                    cntPoints += 1
                elif MBR0[0] > calDistance(p1.x, query.x, p1.y, query.y):
                    ABL.pop(0)
                    cntMBRs += 1
                    cntPoints += 1
                elif calDistance(p1.x, query.x, p1.y, query.y) > calMinMaxDist(MBR0[1].MBR, query):
                    ABL.pop(1)
                    cntPoints += 1
                elif calDistance(p0.x, query.x, p0.y, query.y) > calMinMaxDist(MBR1[1].MBR, query):
                    ABL.pop(0)
                    cntPoints += 1
                else:
                    abl = ABL.pop(0)
                    ABL_new.append(abl)
        ABL += ABL_new
        tupleList += ABL

    if tupleList == []:
        return
    if distance < tupleList[0][0]:
        return
    search(tupleList, query)
    return

# read csv file and process it
POI = pd.read_csv('AllPOI Locatoion Only.csv', delimiter=',', names=["latitude", "longitude"])
points = np.array(POI).tolist()
pointsList = []; id = 0
# store all points and its index into list
for p in points:
    pointsList.append([id] + p)
    id += 1

# Task2, 3: output result
q = Point(0, int(input("x of query:")), int(input("y of query:")))
for n in [10, 100]:
  for d in [2, 6]:
      nodes = {}; id = 0
      print('the bucket size n:{},maximum of d MBRs/subtrees:{}'.format(n, d))
      Rtree = createTree(d, n, pointsList)
      # task2(1)
      print('the height of Rtree is:{}'.format(Rtree.root.height))
      # task2(2)
      non_leaf, leaf = classify_nodes(Rtree)
      print('the number of non-leaf nodes is:{},leaf nodes is:{}'.format(len(non_leaf), len(leaf)))
      # task2(3)
      space_utilisation = count_spaceUtilisation(leaf)
      print('0~25%: {}, 25~50%: {}, 50~75%: {}, 75~100%: {}'.format(space_utilisation[0], space_utilisation[1],
                                                                    space_utilisation[2], space_utilisation[3]))
      # task2(4)
      cntOverlap = MBRoverlap_count(Rtree)
      print('MBR overlap count:{}'.format(cntOverlap))
      # task3
      distance = float('inf')
      results = []
      cntNodes = 0; cntPoints = 0; cntMBRs = 0
      search([(0, Rtree.root)], query = q)
      print('the point p in D that is the nearest neighbour of q:{}'.format(results[0]))
      # task 4
      print('distance:{}, cntNodes:{}, cntPoints:{}, cntMBRs:{}'.format(distance, cntNodes, cntPoints, cntMBRs))
