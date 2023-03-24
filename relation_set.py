import heapq as hq


class RelationSet:
    def __init__(self, relation_type=None):
        relations = {
            1: ['per:schools_attended'], 
            2: ['per:employee_of'], 
            3: ['per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:countries_of_residence'],
            4: ['org:top_members/employees']}
        self.heap = []
        self.set = set()
        self.relation = relations[relation_type]

    def __len__(self):
        return len(self.set)
    
    def __str__(self):
        for i in range(len(self.heap)):
            relation = self.heap[i]
            output += f'{i}) Confidence: {relation[0]}:<{20}| Subject: {relation[1][0]}:<{25}| Object: {relation[1][2]}:<{25}\n'
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