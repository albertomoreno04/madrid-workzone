import osmium, pathlib

class Pass1(osmium.SimpleHandler):
    def __init__(self, w, s, e, n):
        super().__init__()
        self.w, self.s, self.e, self.n = w, s, e, n
        self.kept_nodes = set()
    def node(self, n):
        loc = n.location
        if loc.valid() and self.w <= loc.lon <= self.e and self.s <= loc.lat <= self.n:
            self.kept_nodes.add(n.id)

class Pass2(osmium.SimpleHandler):
    def __init__(self, kept_nodes):
        super().__init__()
        self.kept_nodes = kept_nodes
        self.kept_ways = set()
        self.needed_nodes = set()
    def way(self, w):
        if any(n.ref in self.kept_nodes for n in w.nodes):
            self.kept_ways.add(w.id)
            for n in w.nodes:
                self.needed_nodes.add(n.ref)

class Pass3(osmium.SimpleHandler):
    def __init__(self, writer, needed_nodes, kept_ways):
        super().__init__()
        self.writer = writer
        self.needed_nodes = needed_nodes
        self.kept_ways = kept_ways
    def node(self, n):
        if n.id in self.needed_nodes:
            self.writer.add_node(n)
    def way(self, w):
        if w.id in self.kept_ways:
            self.writer.add_way(w)

src = "data/external/osm/madrid-latest.osm.pbf"
dst = "data/external/osm/madrid_m30_inner.osm"
W, S, E, N = -3.738, 40.395, -3.652, 40.466

print("pass 1: collect bbox nodes")
p1 = Pass1(W, S, E, N); p1.apply_file(src)
print("  nodes in bbox:", len(p1.kept_nodes))

print("pass 2: collect ways that touch bbox")
p2 = Pass2(p1.kept_nodes); p2.apply_file(src)
print("  kept ways:", len(p2.kept_ways), "nodes needed:", len(p2.needed_nodes))

print("pass 3: write XML")
writer = osmium.SimpleWriter(dst)
p3 = Pass3(writer, p2.needed_nodes, p2.kept_ways); p3.apply_file(src)
writer.close()
print("done:", dst, pathlib.Path(dst).stat().st_size, "bytes")
