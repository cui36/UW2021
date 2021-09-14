import heapq


class Cell(object):
    def __init__(self, x, y, reachable):
        self.reachable = reachable
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0


class AStar(object):
    def __init__(self):
        self.op = []
        heapq.heapify(self.op)
        self.cl = set()
        self.cells = []
        self.gridHeight = 10
        self.gridWidth = 10

    def init_grid(self):
        walls = ((2, 4), (2, 5), (2, 6), (3, 6), (4, 6), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2))
        for x in range(self.gridWidth):
            for y in range(self.gridHeight):
                if (x, y) in walls:
                    reachable = False
                else:
                    reachable = True
                self.cells.append(Cell(x, y, reachable))
        self.start = self.get_cell(0, 0)
        self.end = self.get_cell(7, 7)

    def get_heuristic(self, cell):
        return  max(abs(cell.x - self.end.x) , abs(cell.y - self.end.y))

    def get_cell(self, x, y):
        return self.cells[x * self.gridHeight + y]

    def get_adjacent_cells(self, cell):
        cells = []
        if cell.x < self.gridWidth - 1:
            cells.append(self.get_cell(cell.x + 1, cell.y))
        if cell.y > 0:
            cells.append(self.get_cell(cell.x, cell.y - 1))
        if cell.x > 0:
            cells.append(self.get_cell(cell.x - 1, cell.y))
        if cell.y < self.gridHeight - 1:
            cells.append(self.get_cell(cell.x, cell.y + 1))
        return cells

    def display_path(self):
        cell = self.end
        while cell.parent is not self.start:
            cell = cell.parent
            print 'path: cell: %d,%d' % (cell.x, cell.y)

    def update_cell(self, adj, cell):
        adj.g = cell.g + 10
        adj.h = self.get_heuristic(adj)
        adj.parent = cell
        adj.f = adj.h + adj.g

    def process(self):
        heapq.heappush(self.op, (self.start.f, self.start))
        while len(self.op):
            f, cell = heapq.heappop(self.op)
            self.cl.add(cell)
            if cell is self.end:
                self.display_path()
                break
            adj_cells = self.get_adjacent_cells(cell)
            for c in adj_cells:
                if c.reachable:
                    if c in self.cl:
                        if (c.f, c) in self.op:
                            if c.g > cell.g + 10:
                                self.update_cell(c, cell)
                    else:
                        self.update_cell(c, cell)
                        heapq.heappush(self.op, (c.f, c))


if __name__ == "__main__":
    a = AStar()
    a.init_grid()
    a.process()
