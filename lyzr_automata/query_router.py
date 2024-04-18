from lyzr_automata.ai_models.model_base import AIModel
from lyzr_automata.base_literals import RoutingType

import re


def capture_text(input_string):
    # This regex matches alphabetic characters, spaces, underscores, and excludes various types of quotation marks
    regex_pattern = r"[a-zA-Z\s_]+"

    # Find all matches in the input string
    matches = re.findall(regex_pattern, input_string)

    # Join the matches back into a string
    captured_text = "".join(matches)

    return captured_text


class QueryRouter:
    def __init__(
        self,
        model: AIModel,
        additional_instructions=None,
        functions=None,
        tools=None,
        tasks=None,
        router_type=RoutingType.FUNCTION,
    ):
        self.model = model
        self.additional_instructions = additional_instructions
        self.function_meta_map = {
            function.name: {
                "name": function.name,
                "description": function.description,
                "input_details": function.input_details,
                "output_details": function.output_details,
                "function": function,
            }
            for function in functions
        }
        self.tasks = tasks if tasks is not None else []
        self.tools = tools if tools is not None else []
        self.router_type = router_type

    def route(self, query, use_llm=True):
            if self.router_type == RoutingType.FUNCTION:
                return self.function_router(query=query,use_llm=use_llm)

    def _generate_fn_router_prompt(self, function_meta_map, query):
        function_meta_list = list(function_meta_map.values())

        """
        Generates a prompt to identify the most suitable function name based on given descriptions.

        :param function_list: A list of dictionaries, each containing 'name' and 'description' keys for functions.
        :param query :  user query.
        :return: A string containing the customized prompt.
        """
        prompt = (
            f"""
    Given the following list of function names and their descriptions, along with descriptions of inputs and expected outputs, identify the function name that best suits the query. {f" Also, follow these additional instructions: {self.additional_instructions}" if self.additional_instructions is not None else ""}

    - **Function Names and Descriptions:**
    """
            + "\n".join(
                [
                    f"  - **Function Name:** `{func['name']}`\n    - **Description:** This function {func['description']}."
                    for func in function_meta_list
                ]
            )
            + f"""

    - **query** {query}
    
    **Question:**
    Based on query provided, what is the exact name of the function that best suits the requirements? Please provide only the function name. [IMPORTANT!] note the function name should be exactly same as provided.
    """
        )
        return prompt


    def _generate_tk_router_prompt(self, task_meta_map, query):
        task_meta_list = list(task_meta_map.values())

        """
        Generates a prompt to identify the most suitable task name based on given descriptions.

        :param task_meta_list: A list of dictionaries, each containing 'name', 'instructions', 'input_type', and 'output_type' keys for tasks.
        :param query : user query.
        :return: A string containing the customized prompt.
        """
        prompt = (
            f"""
    Given the following list of task names, their instructions, input types, and expected output types, along with a user query, identify the task name that best suits the query. {f" Also, follow these additional instructions: {self.additional_instructions}" if self.additional_instructions is not None else ""}

    - **Task Names and Details:**
    """
            + "\n".join(
                [
                    f"  - **Task Name:** `{task['name']}`\n    - **Instructions:** {task['instructions']}\n    - **Input Type:** {task['input_type']}\n    - **Output Type:** {task['output_type']}."
                    for task in task_meta_list
                ]
            )
            + f"""

    - **Query:** {query}

    **Question:**
    Based on the query provided, what is the exact name of the task that best suits the requirements? Please provide only the task name. [IMPORTANT!] note the task name should be exactly the same as provided.
    """
        )
        return prompt

    def function_router(self, query, use_llm):
        if use_llm:
            prompt = str(
                self._generate_fn_router_prompt(
                    function_meta_map=self.function_meta_map, query=query
                )
            )
            fn_key_name = self.model.generate_text(prompt=prompt)
            return self.function_meta_map[capture_text(fn_key_name)]["function"]
        else:
            for name, meta in self.function_meta_map.items():
                if name in query:
                    return meta["function"]

            for name, meta in self.function_meta_map.items():
                pattern = re.compile(name, re.IGNORECASE)
                if pattern.search(query):
                    return meta["function"]
        return None
