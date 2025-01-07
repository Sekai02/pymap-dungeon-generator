import enum
import heapq
import random
import argparse
import collections
import time

from matplotlib import pyplot
from matplotlib import gridspec
from complete_dungeon_graph import CompleteDungeonGraph
from maximum_spanning_tree import MaximumSpanningTree

parser = argparse.ArgumentParser(description='Generate dungeon.')

parser.add_argument('-f', '--filename', metavar='FILENAME', type=str, default='last.png', help='save result to file')
parser.add_argument('-s', '--show', action='store_true', default=False, help='show result in window')
parser.add_argument('-b', '--blocks', type=int, default=(3, 15), nargs=2, help='blocks in room [min, max]')
parser.add_argument('-r', '--rooms', type=int, default=25, help='rooms in dungeon')
parser.add_argument('-d', '--doors', type=int, default=(2, 4), nargs=2, help='doors in room [min, max]')
parser.add_argument('--show-doors', action='store_true', default=False, help='show doors')

arguments = parser.parse_args()

class DIRECTION(enum.Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

def random_color():
    return '#' + ''.join([random.choice('0123456789') for i in range(6)])


def points_at_circle(x, y, radius):
    points = set()

    for i in range(radius + 1):
        points.add((x + i, y + (radius - i)))
        points.add((x + i, y - (radius - i)))
        points.add((x - i, y + (radius - i)))
        points.add((x - i, y - (radius - i)))

    return points


def restore_path(path_map, point):
    path = []

    while point is not None:
        path.append(point)
        point = path_map[point]

    path.reverse()

    return path


def find_path(point_from, point_to, filled_cells, max_path_length):

    index = 0

    heap = [(0, index, point_from, None)]

    visited_points = {}

    path_map = {}

    while True:

        cost, _, point, prev_point = heapq.heappop(heap)

        path_map[point] = prev_point

        if max_path_length <= cost:
            return None, None

        if point == point_to:
            return cost, restore_path(path_map, point_to)

        visited_points[point] = cost

        for next_point in point.neighbours():
            if next_point in visited_points:
                continue

            if next_point in filled_cells:
                continue

            index += 1
            heapq.heappush(heap, (cost + 1, index, next_point, point))

    return None, None


def make_countur(segments):

    segments = list(segments)

    line = list(segments.pop())

    while True:

        end_point = line[-1]

        for segment in segments:
            if end_point == segment[0]:
                line.append(segment[1])
                segments.remove(segment)
                break
        else:
            break

    return line

class Position:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def neighbours(self):
        return {Position(self.x - 1, self.y),
                Position(self.x, self.y - 1),
                Position(self.x + 1, self.y),
                Position(self.x, self.y + 1)}

    def area(self):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield Position(self.x + dx, self.y + dy)

    def move(self, dx, dy):
        return Position(self.x + dx, self.y + dy)

    def rotate_clockwise(self):
        return Position(self.y, -self.x)

    def point(self):
        return (self.x, self.y)
    
class Border:
    __slots__ = ('position', 'direction', 'internal', 'can_has_door', 'used')

    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.internal = False
        self.can_has_door = False
        self.used = False

    def __eq__(self, other):
        return (self.position, self.direction) == (other.position, other.direction)

    def __ne__(self, other):
        return not self.__eq__(other)

    def mirror(self):
        if self.direction == DIRECTION.LEFT:
            return Border(self.position.move(-1, 0), DIRECTION.RIGHT)

        if self.direction == DIRECTION.RIGHT:
            return Border(self.position.move(1, 0), DIRECTION.LEFT)

        if self.direction == DIRECTION.UP:
            return Border(self.position.move(0, 1), DIRECTION.DOWN)

        if self.direction == DIRECTION.DOWN:
            return Border(self.position.move(0, -1), DIRECTION.UP)

    def geometry_borders(self):
        if self.direction == DIRECTION.LEFT:
            return [self.position.move(0, 0).point(),
                    self.position.move(0, 1).point()]

        if self.direction == DIRECTION.RIGHT:
            return [self.position.move(1, 1).point(),
                    self.position.move(1, 0).point()]

        if self.direction == DIRECTION.UP:
            return [self.position.move(0, 1).point(),
                    self.position.move(1, 1).point()]

        if self.direction == DIRECTION.DOWN:
            return [self.position.move(1, 0).point(),
                    self.position.move(0, 0).point()]

    def move(self, dx, dy):
        self.position = self.position.move(dx, dy)

    def rotate_clockwise(self):
        self.position = self.position.rotate_clockwise()

        if self.direction == DIRECTION.LEFT:
            self.direction = DIRECTION.UP

        elif self.direction == DIRECTION.RIGHT:
            self.direction = DIRECTION.DOWN

        elif self.direction == DIRECTION.UP:
            self.direction = DIRECTION.RIGHT

        elif self.direction == DIRECTION.DOWN:
            self.direction = DIRECTION.LEFT

    def connection_point(self):
        segment = self.geometry_borders()

        return ((segment[0][0] + segment[1][0]) / 2,
                (segment[0][1] + segment[1][1]) / 2)
    
class Block:
    __slots__ = ('position', 'borders')

    def __init__(self, position):
        self.position = position

        self.borders = {DIRECTION.RIGHT: Border(position, DIRECTION.RIGHT),
                        DIRECTION.LEFT: Border(position, DIRECTION.LEFT),
                        DIRECTION.UP: Border(position, DIRECTION.UP),
                        DIRECTION.DOWN: Border(position, DIRECTION.DOWN)}

    def geometry_borders(self):
        return [border.geometry_borders()
                for border in self.borders.values()
                if not border.internal]

    def sync_borders_with(self, block):
        for own_border in self.borders.values():
            for other_border in block.borders.values():
                if own_border.mirror() == other_border:
                    own_border.internal = True
                    other_border.internal = True

    def move(self, dx, dy):
        self.position = self.position.move(dx, dy)

        for border in self.borders.values():
            border.move(dx, dy)

    def rotate_clockwise(self):
        self.position = self.position.rotate_clockwise()

        for border in self.borders.values():
            border.rotate_clockwise()

        self.borders = {border.direction: border for border in self.borders.values()}

class Room:
    __slots__ = ('blocks', 'color')

    def __init__(self):
        self.blocks = [Block(Position(0, 0))]
        self.color = random_color()

    def block_positions(self):
        return {block.position for block in self.blocks}

    def area_positions(self):
        area = set()

        for position in self.block_positions():
            area |= set(position.area())

        return area

    def allowed_new_block_positions(self):
        allowed_positions = set()

        for block in self.blocks:
            allowed_positions |= block.position.neighbours()

        allowed_positions -= self.block_positions()

        return allowed_positions

    def expand(self):
        new_position = random.choice(list(self.allowed_new_block_positions()))

        new_block = Block(new_position)

        for block in self.blocks:
            block.sync_borders_with(new_block)

        self.blocks.append(new_block)

    def geometry_borders(self):
        borders = []

        for block in self.blocks:
            borders.extend(block.geometry_borders())

        return borders

    def rectangle(self):
        positions = self.block_positions()

        min_x, max_x, min_y, max_y = 0, 0, 0, 0

        for position in positions:
            min_x = min(position.x, min_x)
            min_y = min(position.y, min_y)
            max_x = max(position.x, max_x)
            max_y = max(position.y, max_y)

        return min_x, min_y, max_x, max_y

    def has_holes(self):
        min_x, min_y, max_x, max_y = self.rectangle()

        block_positions = self.block_positions()

        all_positions = set()

        # add additional empty cells around rectangle
        # to guaranty connectedness
        for x in range(min_x - 1, max_x + 2):
            for y in range(min_y - 1, max_y + 2):
                all_positions.add(Position(x, y))

        all_positions -= block_positions

        first_position = next(iter(all_positions))

        queue = collections.deque()

        queue.append(first_position)

        while queue:
            position = queue.popleft()

            if position not in all_positions:
                continue

            queue.extend(position.neighbours())

            all_positions.remove(position)

        return bool(all_positions)

    def is_intersect(self, room):
        return bool(self.area_positions() & room.block_positions())

    def move(self, dx, dy):
        for block in self.blocks:
            block.move(dx, dy)

    def rotate_clockwise(self):
        for block in self.blocks:
            block.rotate_clockwise()

    def borders(self):
        for block in self.blocks:
            for border in block.borders.values():
                yield border

    def door_borders(self):
        for border in self.borders():
            if border.can_has_door:
                yield border

    def place_doors(self, number):
        borders = [border
                   for border in self.borders()
                   if not border.internal]

        number = min(len(borders), number)

        for border in random.sample(borders, number):
            border.can_has_door = True

class Corridor:
    __slots__ = ('start_border', 'stop_border', 'path')

    def __init__(self, start_border, stop_border, path):
        self.start_border = start_border
        self.stop_border = stop_border
        self.path = path

    def geometry_segments(self):
        points = [self.start_border.connection_point()]

        points.extend(position.move(0.5, 0.5).point() for position in self.path)

        points.append(self.stop_border.connection_point())

        return points
    
class Dungeon:
    __slots__ = ('rooms', 'corridors', 'mst')

    def __init__(self, mst):
        self.rooms = []
        self.corridors = []
        self.mst = mst

    def create_room(self, blocks, doors):
        room = Room()

        for i in range(random.randint(*blocks)):
            room.expand()

        room.place_doors(random.randint(*doors))

        return room

    def door_borders(self):
        for room in self.rooms:
            for border in room.door_borders():
                if not border.used:
                    yield border

    def is_intersect_room(self, room):
        return any(current_room.is_intersect(room) for current_room in self.rooms)

    def room_positions_bruteforce(self, max_intersection_radius, new_room, dungeon_positions):

        filled_cells = {position.point() for position in dungeon_positions}

        for max_distance in range(0, max_intersection_radius):

            for dungeon_door in self.door_borders():
                for new_room_door in new_room.door_borders():

                    for x, y in points_at_circle(*dungeon_door.position.point(), radius=max_distance):

                        if (x, y) in filled_cells:
                            continue

                        for _ in range(4):
                            new_room.rotate_clockwise()

                            new_room.move(x - new_room_door.mirror().position.x,
                                          y - new_room_door.mirror().position.y)

                            if self.is_intersect_room(new_room):
                                continue

                            yield (max_distance, dungeon_door, new_room_door, x, y)

    def block_positions(self):
        positions = set()

        for room in self.rooms:
            positions |= room.block_positions()

        return positions

    def expand(self, blocks, doors, max_intersection_radius=10):
        new_room = None

        while new_room is None or new_room.has_holes():
            print('try to generate room')
            new_room = self.create_room(blocks=blocks,
                                        doors=doors)

        if len(self.rooms) == 0:
            self.rooms.append(new_room)
            return

        dungeon_positions = self.block_positions()

        corridor_path = None

        # ATTENTION: method room_positions_bruteforce make modifications of new_room
        #            it is not very good decission
        for max_distance, dungeon_door, new_room_door, x, y in self.room_positions_bruteforce(max_intersection_radius,
                                                                                              new_room,
                                                                                              dungeon_positions):
            dungeon_door_out_position = dungeon_door.mirror().position
            new_room_door_out_position = new_room_door.mirror().position

            filled_positions = dungeon_positions | new_room.block_positions()

            path_length, corridor_path = find_path(dungeon_door_out_position,
                                                   new_room_door_out_position,
                                                   filled_cells=filled_positions,
                                                   max_path_length=max_distance)

            if path_length is None:
                continue

            break

        else:
            raise Exception('Can not place room')

        self.rooms.append(new_room)

        # ATTENTION: it is very bad decission, to store objects by links in two different parent objects
        #            beteer solution will be to store threre ID's or something similar
        #new_corridor = Corridor(dungeon_door, new_room_door, corridor_path)

        #self.corridors.append(new_corridor)

    def connect_rooms_via_mst(self):
        if len(self.rooms) <= 1:
            return

        room_positions = [room.block_positions() for room in self.rooms]

        for u, v, _ in self.mst._mst.edges(data=True):
            room_u = self.rooms[u]
            room_v = self.rooms[v]

            #find the closest pair of borders between room_u and room_v
            closest_distance = float('inf')
            best_u_border, best_v_border = None, None

            for border_u in room_u.door_borders():
                for border_v in room_v.door_borders():
                    distance = abs(border_u.position.x - border_v.position.x) + abs(border_u.position.y - border_v.position.y)
                    if distance < closest_distance:
                        closest_distance = distance
                        best_u_border, best_v_border = border_u, border_v

            if best_u_border and best_v_border:
                filled_positions = self.block_positions()

                path_length, corridor_path = find_path(best_u_border.mirror().position,
                                                       best_v_border.mirror().position,
                                                       filled_cells=filled_positions,
                                                       max_path_length=closest_distance + 10)

                if corridor_path:
                    self.corridors.append(Corridor(best_u_border, best_v_border, corridor_path))

def extract_meta_rooms(graph:CompleteDungeonGraph):
    meta_rooms = []

    for node_index in range(len(graph.graph.nodes)):
        meta_rooms.append(graph.graph.nodes[node_index]['room']) 

    return meta_rooms

def draw_legend(dungeon, graph, meta_rooms, mst):
    legend_entries = []

    for index, room in enumerate(dungeon.rooms):
        meta_room = meta_rooms[index]
        meta_tags = f"Theme: {meta_room.theme_tag}, Function: {meta_room.function_tag}, " \
                    f"Difficulty: {meta_room.difficulty_tag}, Environment: {meta_room.environment_tag}"
        legend_entries.append((room.color, meta_tags))

    for color, label in legend_entries:
        pyplot.plot([], [], color=color, label=label, linewidth=5)

    pyplot.legend(loc='upper right', title='Room Legend')


#start_time = time.time()

graph = CompleteDungeonGraph(arguments.rooms)
mst = MaximumSpanningTree(graph)
dungeon = Dungeon(mst)

#end_time = time.time()
#print(f"Time taken for graph creation, MST calculation, and dungeon initialization: {end_time - start_time} seconds")

#debug edges
mst.debug_print_edges()
#debug tree
print("Is tree = ", mst.is_tree())

for i in range(arguments.rooms):
    print('generate room', i + 1)
    dungeon.expand(blocks=arguments.blocks,
                   doors=arguments.doors)
    
dungeon.connect_rooms_via_mst()

pyplot.axes().set_aspect('auto')

fig = pyplot.figure(figsize=(16, 16))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = fig.add_subplot(gs[0])
ax1.set_aspect('auto')

for idx, room in enumerate(dungeon.rooms):
    borders = list(room.geometry_borders())
    ax1.fill(*zip(*make_countur(borders)), '#ffffff')
    ax1.fill(*zip(*make_countur(borders)), room.color, alpha=0.5)

    for border in borders:
        ax1.plot(*zip(*border), color=room.color, linewidth=3, alpha=1.0)

    room_positions = room.block_positions()
    centroid_x = sum(pos.x for pos in room_positions) / len(room_positions)
    centroid_y = sum(pos.y for pos in room_positions) / len(room_positions)

    ax1.text(centroid_x + 0.5, centroid_y + 0.5, str(idx + 1), color='black',
             fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

if arguments.show_doors:
    for room in dungeon.rooms:
        for door_border in room.door_borders():
            ax1.plot(*zip(*door_border.geometry_borders()), color=room.color, linewidth=6, alpha=0.5)

for corridor in dungeon.corridors:
    corridor_color = 'black'
    ax1.plot(*zip(*corridor.geometry_segments()), color=corridor_color, linewidth=3, alpha=1, zorder=0)

ax1.set_title("Dungeon Map")
ax1.axis('on')

ax2 = fig.add_subplot(gs[1])
meta_rooms = extract_meta_rooms(graph)

legend_entries = []
for index, room in enumerate(dungeon.rooms):
    meta_room = meta_rooms[index]
    meta_tags = f"{index + 1}. Theme: {meta_room.theme_tag}, Function: {meta_room.function_tag}, " \
                f"Difficulty: {meta_room.difficulty_tag}, Environment: {meta_room.environment_tag}"
    legend_entries.append((room.color, meta_tags))

for color, label in legend_entries:
    ax2.plot([], [], color=color, label=label, linewidth=5)

ax2.legend(loc='center', title='Room Legend', ncol=2, frameon=False)
ax2.axis('off')

if arguments.filename:
    pyplot.savefig(arguments.filename)

if arguments.show:
    pyplot.show()