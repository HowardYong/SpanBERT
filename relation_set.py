import heapq as hq


class RelationSet:
    def __init__(self):
        self.heap = []
        self.set = set()

    def __len__(self):
        return len(self.set)
    
    def __str__(self):
        output = 'Relation set:\n'
        for relation in self.heap:
            output += f'\tRelation: {relation[1][1]}, Subj: {relation[1][0]}, Obj: {relation[1][2]}\n'
        return output

    def add(self, element, priority):
        if element not in self.set:
            hq.heappush(self.heap, (priority, element))
            self.set.add(element)

    def pop(self):
        priority, element = hq.heappop(self.heap)
        self.set.remove(element)
        return element
        
    def size(self):
        return len(self.set)