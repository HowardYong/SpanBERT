import heapq as hq


class RelationSet:
    def __init__(self, relation_type=None, model='spanbert'):
        relations = {
            1: ['per:schools_attended'], 
            2: ['per:employee_of'], 
            3: ['per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:countries_of_residence'],
            4: ['org:top_members/employees']}
        self.queue = []
        self.set = set()
        self.relation = relations[relation_type]
        self.model = model


    def __len__(self):
        return len(self.queue)
    

    def __str__(self):
        confidence_width = 30
        subject_width = 30
        object_width = 30

        sorted_queue = sorted(self.queue, reverse=True)
        output = ''

        if self.model == 'spanbert':
            print(f"{'Confidence':<{confidence_width}}| {'Subject':<{subject_width}}| {'Object':<{object_width}}\n")
            for i in range(len(sorted_queue)):
                relation = sorted_queue[i]
                output += f"{relation[0]:<{confidence_width}}| {relation[1][0]:<{subject_width}}| {relation[1][2]:<{object_width}}\n"
        elif self.model == 'gpt3':
            print(f"{'Subject':<{subject_width}}| {'Object':<{object_width}}\n")
            for i in range(len(sorted_queue)):
                relation = sorted_queue[i]
                output += f"{relation[1][0]:<{subject_width}}| {relation[1][2]:<{object_width}}\n"
        return output


    def __getitem__(self, i):
        sorted_queue = sorted(self.queue, reverse=True)
        return sorted_queue[i]


    def add(self, element, priority):
        num_dup = 0
        if element in self.set:
            element_idx = next((index for index, (_, rel) in enumerate(self.queue) if element == rel), -1)
            if element_idx != -1 and self.queue[element_idx][0] > priority:
                num_dup += 1
                print(f"\tRelation ({element}) has already been encountered with higher confidence. Skipping...")
            else:
                if self.model != 'gpt3':
                    print(f'\tUpdating confidence for following relation: {self.queue[element_idx][1]}')
                    self.queue[element_idx] = (priority, element)
        else:
            hq.heappush(self.queue, (priority, element))
            self.set.add(element)
        return num_dup

